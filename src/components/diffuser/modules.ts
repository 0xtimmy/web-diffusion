import * as torch from "../../torch"

export class SelfAttention extends torch.nn.Module {

    channels: number;
    size: number;
    mha: torch.nn.MultiheadAttention;
    ln: torch.nn.LayerNorm;
    ff_self: torch.nn.Sequential;

    constructor(channels, size) {
        super();
        this.channels = channels;
        this.size = size;
        this.mha = new torch.nn.MultiheadAttention(channels, 4, true, true);
        this.ln = new torch.nn.LayerNorm([channels]);
        this.ff_self = new torch.nn.Sequential([
            new torch.nn.LayerNorm([channels]),
            new torch.nn.Linear(channels, channels),
            new torch.nn.GeLU(),
            new torch.nn.Linear(channels, channels)
        ])
        
    }

    forward(x: torch.Tensor): torch.Tensor {
        x = x.view([-1, this.channels, this.size * this.size]).transpose(1, 2);
        let q = this.ln.forward(x);
        let k = q.copy();
        let v= q.copy();
        let attention_value = this.mha.forward(q, k, v).output;
        q.destroy();
        k.destroy();
        v.destroy();

        attention_value = attention_value.add(x);
        const ff = this.ff_self.forward(attention_value);
        const output = ff.add(attention_value).transpose(2, 1).view([-1, this.channels, this.size, this.size]);
        attention_value.destroy();

        return output;
    }
}

export class DoubleConv extends torch.nn.Module {

    residual: boolean;
    double_conv: torch.nn.Sequential;

    constructor(in_channels: number, out_channels: number, mid_channels?: number, residual=false) {
        super();
        this.residual = residual;
        if (!mid_channels) {
            mid_channels = out_channels;
        }
        this.double_conv = new torch.nn.Sequential([
            new torch.nn.Conv2d(in_channels, mid_channels, 3, 1, 1, 1, 1, false),
            new torch.nn.GroupNorm(1, mid_channels),
            new torch.nn.GeLU(),
            new torch.nn.Conv2d(mid_channels, out_channels, 3, 1, 1, 1, 1, false),
            new torch.nn.GroupNorm(1, out_channels)
        ])
    }

    forward(x) {
        if (this.residual) {
            return this.double_conv.forward(x).add(x).gelu();
        } else {
            return this.double_conv.forward(x);
        }
    }
}

export class Down extends torch.nn.Module {

    maxpool_conv: torch.nn.Sequential;
    emb_layer: torch.nn.Sequential;

    constructor(in_channels: number, out_channels: number, emb_dim=256) {
        super();
        
        this.maxpool_conv = new torch.nn.Sequential([
            new torch.nn.MaxPool2d(2),
            new DoubleConv(in_channels, in_channels, in_channels, true),
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
        x = this.maxpool_conv.forward(x);
        t = this.emb_layer.forward(t);
        const emb = torch.repeat(t, [1, 1, x.shape[x.shape.length-2], x.shape[x.shape.length-1]]);
        return emb.add(x);
    }
}

export class Up extends torch.nn.Module {

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
        console.log("x shape, skip_x shape: ", x.shape, skip_x.shape);
        x = torch.cat(skip_x, x, 1);
        x = this.conv.forward(x);
        t = this.emb_layer.forward(t);
        let emb = torch.repeat(t, [1, 1, x.shape[x.shape.length-2], x.shape[x.shape.length-1]]);
        return torch.add(x, emb);
    }
}


export class UNet extends torch.nn.Module {
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
        this.outc = new torch.nn.Conv2d(64, c_out, 1);
    }

    pos_encoding(t: torch.Tensor, channels: number) {
        const range = torch.arange(0, channels, 2).scalar_div(channels).scalar_mul(-1);
        const inv_freq = torch.constant(range.shape, 10000).pow(range).unsqueeze(0);
        range.destroy();
        const pos_enc_a = torch.repeat(t, [1, Math.floor(channels / 2)]).mul(inv_freq).sin();
        const pos_enc_b = torch.repeat(t, [1, Math.floor(channels / 2)]).mul(inv_freq).cos();
        inv_freq.destroy();
        const pos_enc = torch.cat(pos_enc_a, pos_enc_b, 1);
        pos_enc_a.destroy();
        pos_enc_b.destroy();
        return pos_enc;
    }

    async forward(x: torch.Tensor, t: torch.Tensor): Promise<torch.Tensor> {
        t = torch.unsqueeze(t, -1);
        const pos_enc = this.pos_encoding(t, this.time_dim);
        
        let x1 = this.inc.forward(x);
        
        const _x2 = this.down1.forward(x1, pos_enc);
        let x2 =  this.sa1.forward(_x2);
        _x2.destroy();
        const _x3 =  this.down2.forward(x2, pos_enc);
        let x3 =  this.sa2.forward(_x3);
        _x3.destroy();
        const _x4 =  this.down3.forward(x3, pos_enc);
        const x4 =  this.sa3.forward(_x4);
        _x4.destroy();
        console.log("finished down");

        const bot1 =  this.bot1.forward(x4);
        x4.destroy();
        const bot2 =  this.bot2.forward(bot1);
        bot1.destroy();
        const bot3 =  this.bot3.forward(bot2);
        bot2.destroy();
        console.log("finished bot")

        const _x5 = this.up1.forward(bot3, x3, pos_enc);
        bot3.destroy();
        x3.destroy();
        const x5 =  this.sa4.forward(_x5);
        _x5.destroy();
        console.log("finished x5");
        const _x6 =  this.up2.forward(x5, x2, pos_enc);
        x5.destroy();
        x2.destroy();
        const x6 =  this.sa5.forward(_x6);
        _x6.destroy();
        console.log("finished x6")
        const _x7 = this.up3.forward(x6, x1, pos_enc);
        x6.destroy();
        x1.destroy();
        const x7 =  this.sa6.forward(_x7);
        _x7.destroy();
        console.log("finished x7")
        
        const x8 = this.outc.forward(x7);
        x7.destroy()
        console.log("finished x8")
        return x8;
    }
}