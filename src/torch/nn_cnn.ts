import { Module } from "./nn_module";
import { Tensor } from "./tensor"
import { Shape } from "./shape";
import * as factories from "./factories"
import * as ops from "./ops"
import { Parameter } from "./nn_module";
import { _calculate_fan_in_fan_out, kaiming_uniform } from "./nn_utils"

export class AvgPooling2d extends Module {}

export class Conv2d extends Module {
    inChannels: number;
    outChannels: number;
    kernelSize: [number, number];
    stride: number | [number, number];
    padding: number | [number, number] | "valid" | "same";
    dilation: number | [number, number];
    groups: number;

    weight: Tensor;
    bias: Parameter;

    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | [number, number],
        stride: number | [number, number] = 1,
        padding: number | [number, number] | "valid" | "same" = 0,
        dilation: number | [number, number] = 1,
        groups=1,
        bias=true,
        padding_mode="zeros",
        dtype: string|null = "float32"
    ) {
        super();

        if(groups <= 0) throw new Error("Conv2d \"groups\" must be a nonzero, positive integrer");
        if(inChannels % groups != 0) throw new Error("Conv2d \"inChannels\" must be evenly divisible by groups");
        if(outChannels % groups != 0) throw new Error("Conv2d \"outChannels\" must be evenly divisible by groups");

        this.inChannels = inChannels;
        this.outChannels = outChannels;

        this.kernelSize = typeof(kernelSize) == 'number' ? [kernelSize, kernelSize] : kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        
        this.weight = new Parameter(factories.empty([outChannels, Math.floor(inChannels / groups), this.kernelSize[0], this.kernelSize[1]]))
        if(bias) this.bias = new Parameter(factories.empty([outChannels]));

        this.reset_parameters();
    }

    reset_parameters() {
        this.weight = new Parameter(kaiming_uniform(this.weight, Math.sqrt(5)));
        if(this.bias) {
            const { fan_in, fan_out } = _calculate_fan_in_fan_out(this.weight);
            if(fan_in != 0) {
                const bound = 1 / Math.sqrt(fan_in);
                this.bias = new Parameter(factories.uniform(this.bias.shape, -bound, bound));
            } else {
                this.bias = new Parameter(factories.zeros(this.bias.shape));
            }

        }
    }

    forward(input: Tensor): Tensor {
        return ops.conv2d(input, this.weight, this.bias, this.stride, this.padding, this.dilation, this.groups);

    }
}

export class ConvTranspose2d extends Module {}

export class LayerNorm extends Module {

    normalized_shape: Shape;
    eps: number;
    elementwise_affine: boolean;

    weight: Parameter;
    bias: Parameter;

    constructor(normalized_shape: Shape, eps=0.00001, elementwise_affine=true) {
        super();
        this.normalized_shape = normalized_shape;
        this.eps = eps;
        this.elementwise_affine=elementwise_affine;

        if(elementwise_affine) {
            this.weight = new Parameter(factories.empty(normalized_shape));
            this.bias = new Parameter(factories.empty(normalized_shape));
        }

        this.reset_parameters();
    }

    reset_parameters() {
        if(this.weight) this.weight = new Parameter(factories.ones(this.weight.shape));
        if(this.bias) this.bias = new Parameter(factories.zeros(this.bias.shape));
    }

    forward(input: Tensor): Tensor {
        return ops.layernorm(input, this.normalized_shape, this.weight, this.bias, this.eps);
    }
}

export class GroupNorm extends Module {
    numGroups: number;
    numChannels: number;
    eps: number;
    affine: boolean;

    weight: Parameter;
    bias: Parameter;

    constructor(numGroups: number, numChannels: number, eps=1e-5, affine=true) {
        super();
        if(numChannels % numGroups != 0) throw new Error(`numChannels must be divisible by numGroups but got numChannels = ${numChannels} and numGroups = ${numGroups}`);
        this.numGroups = numGroups;
        this.numChannels = numChannels;

        this.eps = eps;
        this.affine = affine;

        if(this.affine) {
            this.weight = new Parameter(factories.empty(numChannels));
            this.bias = new Parameter(factories.empty(numChannels));
        } else {
            this.registerParameter("weight", null);
            this.registerParameter("bias", null);
        }

        this.reset_parameters();
    }

    reset_parameters() {
        if(this.affine) {
            this.weight = new Parameter(factories.ones(this.numChannels));
            this.bias = new Parameter(factories.zeros(this.numChannels));
        }
    }

    forward(input: Tensor): Tensor {
        //this.reset_parameters();
        return ops.group_norm(input, this.numGroups, this.weight, this.bias, this.eps);
    }
}

export class Linear extends Module {
    inChannels: number;
    outChannels: number;

    weight: Parameter;
    bias: Parameter;

    constructor(inChannels: number, outChannels: number, bias=true) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        this.weight = new Parameter(factories.empty([outChannels, inChannels]));
        if(bias) this.bias = new Parameter(factories.empty(outChannels));

        this.reset_parameters();
    }

    reset_parameters() {
        const k = Math.sqrt(1 / this.inChannels);
        this.weight = new Parameter(factories.uniform([this.outChannels, this.inChannels], -k, k));
        if(this.bias) this.bias = new Parameter(factories.uniform([this.outChannels], -k, k));
    }

    forward(input: Tensor): Tensor {
        //this.reset_parameters();
        return ops.linear(input, this.weight, this.bias);
    }
}
