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
            print(res.shape)
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
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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