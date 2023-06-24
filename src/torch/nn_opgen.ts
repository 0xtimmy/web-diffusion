import { Tensor } from "./tensor";
import { Module, Parameter } from "./nn_module";
import { Shape } from "./shape";
import { empty, constant } from "./factories";
import { xavier_uniform, xavier_normal } from "./nn_utils";
import { Linear } from "./nn_cnn";
import * as aops from "./ops_artisanal";
// ------------------------------------
// Start Custom
// ------------------------------------

export class GeLU extends Module {
    forward(input: Tensor): Tensor {
        //(async () => { console.log("forwarding GeLU with input: ", await input.toArrayAsync()); })();
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
        //(async () => { console.log("forwarding LayerNorm with input: ", await input.toArrayAsync()); })();
        return aops.layernorm(input, this.normalized_shape, undefined, undefined, this.eps);
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

    add_zero_attn: boolean;

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
        this.out_proj = new Linear(embed_dim, embed_dim);

        if(add_bias_kv) {
            this.bias_k = new Parameter(empty([1, 1, embed_dim]));
            this.bias_v = new Parameter(empty([1, 1, embed_dim]));
        }

        this.add_zero_attn = add_zero_attn;

        this._reset_parameters();
    }

    _reset_parameters() {
        if(this._qkv_same_embed_dim) {
            this.in_proj_weight = xavier_uniform(this.in_proj_weight);
        } else {
            this.q_proj_weight = xavier_uniform(this.q_proj_weight);
            this.k_proj_weight = xavier_uniform(this.k_proj_weight);
            this.v_proj_weight = xavier_uniform(this.v_proj_weight);
        }

        if(this.in_proj_bias != null) {
            this.in_proj_bias = constant(this.in_proj_bias.shape, 0.0);
            this.out_proj.bias = constant(this.out_proj.bias.shape, 0.0);
        }
        if(this.bias_k != null) this.bias_k = xavier_normal(this.bias_k);
        if(this.bias_v != null) this.bias_v = xavier_normal(this.bias_v);
    }

    forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | null = null,
        need_weights=true,
        attn_mask: Tensor | null = null,
        average_attn_weights=true,
        is_casual=false
    ): { output: Tensor, weights: Tensor } {

        //(async () => { console.log("forwarding multihead attention with input: ", await query.toArrayAsync()); })();
        const is_batched = query.dim == 3;

        /*
        key_padding_mask = canonical_mask(
            key_padding_mask,
            "key_padding_mask",
            attn_mask ? attn_mask.dtype : null,
            "attn_mask",
            query.dtype
        ); 

        attn_mask = canonical_mask(
            attn_mask,
            "attn_mask",
            null,
            "",
            query.dtype,
            false
        );
        */

        let why_not_fast_path = "";
        if(!is_batched) why_not_fast_path = `input not batched, expexted dim of 3 but got ${query.dim}`;
        else if(query != key || key != value) why_not_fast_path = `non self attention used`;
        else if (this.in_proj_bias != null && query.dtype != this.in_proj_bias.dtype) why_not_fast_path = `dtypes of query (${query.dtype}) and self.in_proj_bias (${this.in_proj_bias.dtype}) don't match`
        else if(this.in_proj_weight == null) why_not_fast_path = "in_proj_weight was null"
        else if(query.dtype != this.in_proj_weight.dtype) why_not_fast_path = `dtypes of query (${query.dtype}) and self.in_proj_weight (${this.in_proj_weight.dtype}) don't match`
        else if(this.training) why_not_fast_path = "training is enabled";
        else if(this.num_heads % 2 != 0) why_not_fast_path = "num heads is not even";
        else if(!this.batch_first) why_not_fast_path = "batch_first was false";
        else if(this.bias_k != null) why_not_fast_path = "bias_k was nt null";
        else if(this.bias_v != null) why_not_fast_path = "bias_v was nt null";
        else if(this.add_zero_attn) why_not_fast_path = "add_zero_attn was enabled";
        else if(!this._qkv_same_embed_dim) why_not_fast_path = "qkv_same_embed_dim was not true";

        let res;
        

        if (!this._qkv_same_embed_dim) {
            res = aops.multihead_attention(
                query, key, value, this.embed_dim, this.num_heads,
                this.in_proj_weight, this.in_proj_bias,
                this.add_zero_attn,
                this.dropout, this.out_proj.weight, this.out_proj.bias,
                undefined,
                undefined,
                true,
                this.q_proj_weight,
                this.k_proj_weight,
                this.v_proj_weight
                /*
                training=this.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal
                */
            )
        } else {
            res = aops.multihead_attention(
                query, key, value, this.embed_dim, this.num_heads,
                this.in_proj_weight, this.in_proj_bias,
                this.add_zero_attn,
                this.dropout, this.out_proj.weight, this.out_proj.bias,
                undefined,
                undefined,
                false,
                this.q_proj_weight,
                this.k_proj_weight,
                this.v_proj_weight
                /*
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal
                */
            )
        }
            
        throw new Error("finished multihead attention");
        //return { output: res.output, weights: res.weights };
        //return input.multihead_attention(this.embed_dim, this.num_heads);
    }
}

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
