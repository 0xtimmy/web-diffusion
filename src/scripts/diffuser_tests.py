import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model_loading import gen_state_dict

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            res = self.double_conv(x)
            return res
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            )
        )
    
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            )
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        print(x.shape, " forwarding x")
        print(t.shape, " with t")
        t = self.pos_encoding(t, self.time_dim)
        print(t.shape, "pos_encoding")

        x1 = self.inc(x)
        print(x1.shape, "inc")
        x2 = self.down1(x1, t)
        print(x2.shape, "down1")
        x2 = self.sa1(x2)
        print(x2.shape, "sa1")
        x3 = self.down2(x2, t)
        print(x3.shape, "down2")
        x3 = self.sa2(x3)
        print(x3.shape, "sa2")
        x4 = self.down3(x3, t)
        print(x4.shape, "down3")
        x4 = self.sa3(x4)
        print(x4.shape, "sa3")

        x4 = self.bot1(x4)
        print(x4.shape, "bot1")
        x4 = self.bot2(x4)
        print(x4.shape, "bot2")
        x4 = self.bot3(x4)
        print(x4.shape, "bot3")

        x = self.up1(x4, x3, t)
        print(x.shape, "up1")
        x = self.sa4(x)
        print(x.shape, "sa4")
        x = self.up2(x, x2, t)
        print(x.shape, "up2")
        x = self.sa5(x)
        print(x.shape, "sa5")
        x = self.up3(x, x1, t)
        print(x.shape, "up3")
        x = self.sa6(x)
        print(x.shape, "sa6")

        output = self.outc(x)
        print(output.shape, "outc")
        return output


def gen_ddpm(message, noise_steps, log="always", log_config="fail"):
    model = UNet()
    ckpt = torch.load("./src/scripts/pokemon/ckpt0.pt", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)

    _beta = torch.linspace(1e-4, 0.02, noise_steps)
    _alpha = 1. - _beta
    _alpha_hat = torch.cumprod(_alpha, dim=0)

    input = None
    noises = []
    results = []
    
    for i in range(noise_steps):
        noises.append(None)
        results.append(None)

    model.eval()
    with torch.no_grad():
        x = torch.randn((1, 3, 64, 64))
        input = x.detach().numpy().tolist()
        for i in reversed(range(1, noise_steps)):
            t = (torch.ones(1) * i).long()
            
            predicted_noise = model(x, t)
            
            alpha = _alpha[t][:, None, None, None]
            alpha_hat = _alpha_hat[t][:, None, None, None]
            beta = _beta[t][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            noises[i] = noise.detach().numpy().tolist()
            
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise
            result = (x.clamp(-1, 1) + 1) / 2
            result = torch.cat([result, torch.ones(1, 1, x.shape[-2], x.shape[-1])], 1)
            result = (result * 255).type(torch.uint8)
            result = torch.cat([
                torch.cat([i for i in result.cpu()], dim=-1),
            ], dim=-2).permute(1, 2, 0)
            results[i] = result.detach().numpy().tolist()
            
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)

    return {
        "message": message,
        "func": "ddpm",
        "args": {
            "input": input,
            "noise_steps": noise_steps,
            "noises": noises,
            "results": results
        },
        "target": x.detach().numpy().tolist(),
        "log": log,
        "log_config": log_config
    }

def gen_pos_enc(message, input, channels, log="always", log_config="fail"):
    
    start = time.time()
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, channels, 2).float() / channels)
    )
    pos_enc_a = torch.sin(input.repeat(1, channels // 2) * inv_freq)
    pos_enc_b = torch.cos(input.repeat(1, channels // 2) * inv_freq)
    output = torch.cat([pos_enc_a, pos_enc_b], dim=1)

    duration = time.time() - start

    return {
        "message": message,
        "func": "pos_enc",
        "args": {
            "input": input.numpy().tolist(),
            "channels": channels,
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_unet(message, input, t, log="always", log_config="fail"):
    model = UNet()
    ckpt = torch.load("./src/scripts/pokemon/ckpt0.pt", map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    start = time.time()

    _t = t.unsqueeze(-1).type(torch.float)
    pos_enc = model.pos_encoding(_t, model.time_dim)

    inc = model.inc(input)
    down1 = model.down1(inc, pos_enc)
    sa1 = model.sa1(down1)
    down2 = model.down2(sa1, pos_enc)
    sa2 = model.sa2(down2)
    down3 = model.down3(sa2, pos_enc)
    sa3 = model.sa3(down3)

    bot1 = model.bot1(sa3)
    bot2 = model.bot2(bot1)
    bot3 = model.bot3(bot2)

    up1 = model.up1(bot3, sa2, pos_enc)
    sa4 = model.sa4(up1)
    up2 = model.up2(sa4, sa1, pos_enc)
    sa5 = model.sa5(up2)
    up3 = model.up3(sa5, inc, pos_enc)
    sa6 = model.sa6(up3)

    output = model.outc(sa6) 

    duration = time.time() - start
    return {
        "message": message,
        "func": "unet",
        "args": {
            "input": input.numpy().tolist(),
            "t": t.numpy().tolist(),
            "pos_enc": pos_enc.numpy().tolist(),
            "inc": inc.detach().numpy().tolist(),
            "down1": down1.detach().numpy().tolist(),
            "sa1": sa1.detach().numpy().tolist(),
            "down2": down2.detach().numpy().tolist(),
            "sa2": sa2.detach().numpy().tolist(),
            "down3": down3.detach().numpy().tolist(),
            "sa3": sa3.detach().numpy().tolist(),
            "bot1": bot1.detach().numpy().tolist(),
            "bot2": bot2.detach().numpy().tolist(),
            "bot3": bot3.detach().numpy().tolist(),
            "up1": up1.detach().numpy().tolist(),
            "sa4": sa4.detach().numpy().tolist(),
            "up2": up2.detach().numpy().tolist(),
            "sa5": sa5.detach().numpy().tolist(),
            "up3": up3.detach().numpy().tolist(),
            "sa6": sa6.detach().numpy().tolist(),
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_self_attention(message, channels, size, input, log="always", log_config="fail"):
    start = time.time()
    sa = SelfAttention(channels, size)
    output = sa(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "self_attention",
        "args": {
            "channels": channels,
            "size": size,
            "input": input.detach().numpy().tolist(),
            "state_dict": gen_state_dict(sa)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_double_conv(message, in_channels, out_channels, mid_channels, residual, input, log="always", log_config="fail"):
    start = time.time()
    conv = DoubleConv(in_channels, out_channels, mid_channels, residual)
    output = conv(input)
    duration = time.time() - start
    return {
        "message": message,
        "func": "double_conv",
        "args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "mid_channels": mid_channels,
            "residual": residual,
            "input": input.detach().numpy().tolist(),
            "state_dict": gen_state_dict(conv)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_down(message, in_channels, out_channels, input, t, log="always", log_config="fail"):
    start = time.time()
    down = Down(in_channels, out_channels)
    output = down(input, t)
    duration = time.time() - start
    return {
        "message": message,
        "func": "down",
        "args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input": input.numpy().tolist(),
            "t": t.numpy().tolist(),
            "state_dict": gen_state_dict(down)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }

def gen_up(message, in_channels, out_channels, input, skip, t, log="always", log_config="fail"):
    start = time.time()
    up = Up(in_channels, out_channels)
    output = up(input, skip, t)
    duration = time.time() - start
    return {
        "message": message,
        "func": "up",
        "args": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input": input.numpy().tolist(),
            "skip": skip.numpy().tolist(),
            "t": t.numpy().tolist(),
            "state_dict": gen_state_dict(up)
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }


def gen_denoise(message, input, i, log="always", log_config="fail"):
    
    
    _beta = torch.linspace(1e-4, 0.02, 1000)
    _alpha = 1. - _beta
    _alpha_hat = torch.cumprod(_alpha, dim=0)

    t = (torch.ones(1) * i).long()

    original_noise = torch.randn((1, 3, 64, 64))

    alpha = _alpha[t][:, None, None, None]
    alpha_hat = _alpha_hat[t][:, None, None, None]
    beta = _beta[t][:, None, None, None]

    if i > 1:
        noise = torch.randn(original_noise.shape)
    else:
        noise = torch.zeros(original_noise.shape)

    start = time.time()
    output = 1 / torch.sqrt(alpha) * (original_noise - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * input) + torch.sqrt(beta) * noise
    duration = time.time() - start

    return {
        "message": message,
        "func": "denoise",
        "args": {
            "input": input.numpy().tolist(),
            "i": i,
            "original_noise": original_noise.numpy().tolist(),
            "noise": noise.numpy().tolist(),
            "alpha": _alpha.numpy().tolist(),
            "alpha_hat": _alpha_hat.numpy().tolist(),
            "beta": _beta.numpy().tolist(),
        },
        "target": output.detach().numpy().tolist(),
        "duration": duration * 1000,
        "log": log,
        "log_config": log_config
    }