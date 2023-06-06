import { Tensor } from "./tensor";
import { Module } from "./nn_module";
import { Shape } from "./shape";
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
