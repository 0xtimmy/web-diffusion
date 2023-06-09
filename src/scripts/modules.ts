import * as torch from "@/torch"
import { arange } from "./utils";
import { Conv2d } from "webgpu-torch/nn_cnn";

class SelfAttention extends torch.nn.Module {

    channels: number;
    size: number;
    mha: torch.nn.Module;
    ln: torch.nn.LayerNorm;
    ff_self: torch.nn.Sequential;

    constructor(channels, size) {
        super();
        this.channels = channels;
        this.size = size;
        this.mha = new torch.nn.MultiheadAttention(channels, 4, );
        this.ln = new torch.nn.LayerNorm([channels]);
        this.ff_self = new torch.nn.Sequential([
            new torch.nn.LayerNorm([channels]),
            new torch.nn.Linear(channels, channels),
            new torch.nn.GeLU(),
            new torch.nn.Linear(channels, channels)
        ])
        
    }

    forward(x) {
        x = x.view(-1, this.channels, this.size + this.size).swapaxes(1, 2);
        let x_ln = this.ln.forward(x);
        let attention_value, _ = this.mha.forward(x_ln, x_ln, x_ln);
        attention_value = attention_value + x;
        attention_value = this.ff_self.forward(attention_value) + attention_value;
        return attention_value.swapaxes(2, 1).view(-1, this.channels, this.size, this.size);

    }
}

class DoubleConv extends torch.nn.Module {

    residual: boolean;
    double_conv: torch.nn.Sequential;

    constructor(in_channels: number, out_channels: number, mid_channels=NaN, residual=false) {
        super();
        this.residual = residual;
        if (isNaN(mid_channels)) {
            mid_channels = out_channels;
        }
        this.double_conv = new torch.nn.Sequential([
            new torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1, "float32"),
            new torch.nn.GroupNorm(1, mid_channels),
            new torch.nn.GeLU(),
            new torch.nn.Conv2d(mid_channels, out_channels, 3, 1, 1, "float32"),
            new torch.nn.GroupNorm(1, out_channels)
        ])
    }

    forward(x) {
        if (this.residual) {
            return torch.gelu(torch.add(x, this.double_conv.forward(x)));
        } else {
            return this.double_conv.forward(x);
        }
    }
}

class Down extends torch.nn.Module {

    maxpool_conv: torch.nn.Sequential;
    emb_layer: torch.nn.Sequential;

    constructor(in_channels: number, out_channels: number, emb_dim=256) {
        super();
        this.maxpool_conv = new torch.nn.Sequential([
            new torch.nn.MaxPool2d(2),
            new DoubleConv(in_channels, in_channels, undefined, true),
            new DoubleConv(in_channels, out_channels)
        ]);

        this.emb_layer = new torch.nn.Sequential([
            new torch.nn.SiLU(),
            new torch.nn.Linear(
                emb_dim,
                out_channels
            )
        ])
    }

    forward(x: torch.Tensor, t: torch.Tensor): torch.Tensor {
        x = this.maxpool_conv.forward(x)
        // scuffed
        const emb = torch.repeat(this.emb_layer.forward(t), [1, 1, x.shape[-2], x.shape[-1]]);
        return torch.add(x, emb);
    }
}

class Up extends torch.nn.Module {

    up: torch.nn.UpSample;
    conv: torch.nn.Sequential;
    emb_layer: torch.nn.Sequential;

    constructor(in_channels: number, out_channels: number, emb_dim=256) {
        super();

        this.up = new torch.nn.UpSample(null, 20, "bilinear");
        this.conv = new torch.nn.Sequential([
            new DoubleConv(in_channels, in_channels, undefined, true),
            new DoubleConv(in_channels, out_channels, Math.floor(in_channels / 2)),
        ])

        this.emb_layer = new torch.nn.Sequential([
            new torch.nn.SiLU(),
            new torch.nn.Linear(emb_dim, out_channels),
        ])
    }

    forward(x, skip_x, t) {
        x = this.up.forward(x);
        x = torch.cat(skip_x, x, 1);
        x = this.conv.forward(x);
        let emb = torch.repeat(this.emb_layer.forward(t), [1, 1, x.shape[-2], x.shape[-1]]);
        return torch.add(x, emb);
    }
}


class UNet extends torch.nn.Module {
    time_dim: number;

    inc: DoubleConv;
    down1: Down;
    sa1: SelfAttention;
    down2: Down;
    sa2: SelfAttention;
    down3: Down;
    sa3: SelfAttention;

    bot1: DoubleConv;
    bot2: DoubleConv;
    bot3: DoubleConv;

    up1: Up;
    sa4: SelfAttention;
    up2: Up;
    sa5: SelfAttention;
    up3: Up;
    sa6: SelfAttention;
    outc: torch.nn.Conv2d;

    constructor(c_in=3, c_out=3, time_dim=256) {
        super();
        this.time_dim = time_dim;
        this.inc = new DoubleConv(c_in, 64);
        this.down1 = new Down(64, 128);
        this.sa1 = new SelfAttention(128, 32);
        this.down2 = new Down(128, 256);
        this.sa2 = new SelfAttention(256, 16);
        this.down3 = new Down(256, 256);
        this.sa3 = new SelfAttention(256, 8);

        this.bot1 = new DoubleConv(256, 512);
        this.bot2 = new DoubleConv(512, 512);
        this.bot3 = new DoubleConv(512, 256);

        this.up1 = new Up(512, 128);
        this.sa4 = new SelfAttention(128, 16);
        this.up2 = new Up(256, 64);
        this.sa5 = new SelfAttention(64, 32);
        this.up3 = new Up(128, 64);
        this.sa6 = new SelfAttention(64, 64);
        this.outc = new torch.nn.Conv2d(64, c_out, 1, 1, 0, "None");
    }

    pos_encoding(t: torch.Tensor, channels: number) {
        const range = torch.scalar_div(arange(0, channels, 2), channels);
        const inv_freq = torch.pow(
            torch.scalar_div(torch.ones(range.shape), 10000),
            range
        );
        const pos_enc_a = torch.sin(torch.mul(torch.repeat(t, 1, Math.floor(channels / 2)), inv_freq));
        const pos_enc_b = torch.cos(torch.mul(torch.repeat(t, 1, Math.floor(channels / 2)), inv_freq));
        const pos_enc = torch.cat(pos_enc_a, pos_enc_b, 1);
        return pos_enc;
    }

    forward(x: torch.Tensor, t: torch.Tensor) {
        //t = torch.unsqueeze(t, -1);
        t = this.pos_encoding(t, this.time_dim);

        let x1 = this.inc(x);
        let x2 = this.down1(x1, t);
        x2 = this.sa1(x2)
        let x3 = this.down2(x2, t);
        x3 = this.sa2(x3);
        let x4 = this.down3(x3, t);
        x4 = this.sa3(x4);

        x4 = this.bot1(x4);
        x4 = this.bot2(x4);
        x4 = this.bot3(x4);

        x = this.up1(x4, x3, t);
        x = this.sa4(x);
        x = this.up2(x, x2, t);
        x = this.sa5(x);
        x = this.up3(x, x1, t);
        x = this.sa6(x)

        const output = this.outc(x)
        return output

    }
}