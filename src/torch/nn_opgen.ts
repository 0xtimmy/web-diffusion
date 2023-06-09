import { Tensor } from "./tensor";
import { Module, Parameter } from "./nn_module";
import { Shape } from "./shape";
import { empty } from "./factories";
// ------------------------------------
// Start Custom
// ------------------------------------

export class GeLU extends Module {
    forward(input: Tensor): Tensor {
        return input.gelu();
    }
}

export class LayerNorm extends Module {

    normalized_shape: Shape;
    eps: number;
    elementwise_affine: boolean;

    constructor(normalized_shape: Shape, eps=0.00001, elementwise_affine=true) {
        super();
        this.normalized_shape = normalized_shape;
        this.eps = eps;
        this.elementwise_affine=elementwise_affine;
    }

    forward(input: Tensor): Tensor {
        return input.layernorm(this.normalized_shape, this.eps);
    }
}

export class MaxPool2d extends Module {

    kernel_size: [number, number];
    stride: [number, number];
    padding: [number, number];
    dilation: [number, number];

    constructor(
        kernel_size: number | [number, number], 
        stride: null | number | [number, number] = null,
        padding: number | [number, number] = 0,
        dilation: number | [number, number] = 1
    ) {
        super();
        this.kernel_size = typeof(kernel_size) == "number" ? [kernel_size, kernel_size] : kernel_size;
        this.stride = stride == null ? this.kernel_size : typeof(stride) == "number" ? [stride, stride] : stride;
        this.padding = typeof(padding) == "number" ? [padding, padding] : padding;
        this.dilation = typeof(dilation) == "number" ? [dilation, dilation] : dilation;
    }

    forward(input: Tensor): Tensor {
        return input.maxpool2d(this.kernel_size, this.stride, this.padding, this.dilation);
    }
}

export class UpSample extends Module {
    
    size: number | [number, number] | [number, number, number] | null;
    scale_factor: number | [number, number] | [number, number, number] | null;
    mode: "nearest" | "linear" | "bilinear" | "bicubic" | "trilinear";
    align_corners: boolean;
    recompute_scale_factor: boolean;

    constructor(
        size: number | [number, number] | [number, number, number] | null = null,
        scale_factor: number | [number, number] | [number, number, number] | null = null,
        mode: "nearest" | "linear" | "bilinear" | "bicubic" | "trilinear" = "nearest",
        align_corners = false,
        recompute_scale_factor = false
    ) {
        super();
        this.size = size;
        this.scale_factor = scale_factor;
        this.mode = mode;
        this.align_corners = align_corners;
        this.recompute_scale_factor = recompute_scale_factor;
    }

    forward(input: Tensor): Tensor {
        return input.upsample(this.size, this.scale_factor, this.mode, this.align_corners, this.recompute_scale_factor);
    }
}
/*
export class MultiheadAttention extends Module {

    embed_dim: number;
    kdim: number;
    vdim: number;
    _qkv_same_embed_dim: boolean;

    num_heads: number;
    dropout: number;
    batch_first: boolean;
    head_dim: number;

    q_proj_weight: Parameter;
    k_proj_weight: Parameter;
    v_proj_weight: Parameter;
    in_proj_weight: Parameter;

    out_proj: any;

    in_proj_bias: Parameter | null = null;
    bias_k: Parameter | null = null;
    bias_v: Parameter | null = null;

    add_zer_attn: boolean;

    constructor(embed_dim: number, num_heads: number, dropout=0, bias=true, add_bias_kv=false, add_zero_attn=false, kdim=null, vdim=null, batch_first=false) {
        super();
        this.embed_dim = embed_dim;
        this.kdim = kdim ? kdim : embed_dim;
        this.vdim = vdim ? vdim : embed_dim;
        this._qkv_same_embed_dim = this.kdim == embed_dim && this.vdim == embed_dim; 

        this.num_heads = num_heads;
        this.dropout = dropout;
        this.batch_first = batch_first;
        this.head_dim = Math.floor(embed_dim / num_heads);
        if(this.head_dim * num_heads != this.embed_dim) throw new Error("embed_dim must be divisible by num_heads");

        if(!this._qkv_same_embed_dim) {
            this.q_proj_weight = new Parameter(empty([this.embed_dim, this.embed_dim]));
            this.k_proj_weight = new Parameter(empty([this.embed_dim, this.kdim]));
            this.v_proj_weight = new Parameter(empty([this.embed_dim, this.vdim]));
            this.registerParameter('in_proj_weight', null);
        } else {
            this.in_proj_weight = new Parameter(empty([3*embed_dim, embed_dim]));
            this.registerParameter('q_proj_weight', null);
            this.registerParameter('k_proj_weight', null);
            this.registerParameter('v_proj_weight', null);
        }
        
        if(bias) this.in_proj_bias = new Parameter(empty(3 * embed_dim));
        else this.registerParameter("in_proj_bias", null);
        this.out_proj = NotDynamicallyQuantizableLinear(embed_dim, embed_dim, bias);

        if(add_bias_kv) {
            this.bias_k = new Parameter(empty([1, 1, embed_dim]));
            this.bias_v = new Parameter(empty([1, 1, embed_dim]));
        }

        this.add_zer_attn = add_zero_attn;

        this._reset_parameters();

        function _reset_paramteres() {
            if(this._qkv_same_embed_dim) {
                xavier_uniform(this.in_proj_weight);
            } else {
                xavier_uniform(this.q_proj_weight);
                xavier_uniform(this.k_proj_weight);
                xavier_uniform(this.v_proj_weight);
            }

            if(this.in_proj_bias != null) {
                constant_(this.in_proj_bias, 0.0);
                constant_(this.out_proj.bias, 0.0);
            }
            if(this.bias_k != null) xavier_normal(this.bias_k);
            if(this.bias_v != null) xavier_normal(this.bias_v);
        }
    }

    forward(input: Tensor): Tensor {
        return input;
        //return input.multihead_attention(this.embed_dim, this.num_heads);
    }
}
*/

// ------------------------------------
// Start Custom
// ------------------------------------


/**
* ![Plot of relu and its gradient](/plots/relu.svg)
*
* Calculates:
* ```js
* output = max(input, 0.0)
* ```
*
* Gradient:
* ```js
* inputGrad = input > 0.0 ? outputGrad : 0.0
* ```
*
*/
export class ReLU extends Module {
    forward(input: Tensor): Tensor {
        return input.relu();
    }
}
/**
* ![Plot of silu and its gradient](/plots/silu.svg)
*
* Calculates:
* ```js
* output = input / (1.0 + exp(-input))
* ```
*
* Gradient:
* ```js
* var out = 1.0 / (1.0 + exp(-input)); inputGrad = outputGrad * (out + input * out * (1.0 - out))
* ```
*
*/
export class SiLU extends Module {
    forward(input: Tensor): Tensor {
        return input.silu();
    }
}
