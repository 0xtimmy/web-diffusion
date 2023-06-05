import type { Deviceish } from "./device";
import type { Dtype } from "./dtype";
import { Tensor } from "./tensor";
import { shouldCreateGradient } from "./autograd";
import type { TensorData, TensorSpec } from "./tensor";
import { Shape } from "./shape";
import { ones } from "./factories";
import { scalar_mul } from "./ops_opgen";

// ------------------------------------
// Start Custom
// ------------------------------------
export function cat(a: Tensor, b: Tensor, dim: 0|1|2|3): Tensor {
    //throw new Error("cat not implemented yet");
    let part: number;
    let stride: number;
    const ashape = [1, 1, 1, 1].map((v, i) => { return typeof a.shape[i] == 'number' ? a.shape[i] : v});
    const bshape = [1, 1, 1, 1].map((v, i) => { return typeof b.shape[i] == 'number' ? b.shape[i] : v});
    if(dim == 0) {
        part = ashape[0] * ashape[1] * ashape[2] * ashape[3];
        stride = part + bshape[0] * bshape[1] * bshape[2] * bshape[3];
    } else if(dim == 1) {
        part = ashape[1] * ashape[2] * ashape[3];
        stride = part + bshape[1] * bshape[2] * bshape[3];
    } else if(dim == 2) {
        part = ashape[2] * ashape[3];
        stride = part + bshape[2] * bshape[3];
    } else if(dim == 3) {
        part = ashape[3];
        stride = part + bshape[3];
    } else {
        throw new Error("Expected dim in range [0, 3]");
    }
    
                 
    const params = {
        dim: dim,
        part: part,
        stride: stride,
        outputSize: ashape[0] * ashape[1] * ashape[2] * ashape[3] + bshape[0] * bshape[1] * bshape[2] * bshape[3]
    };
    const output_shape = ashape.map((size: number, i: number) => {
        if(i == dim) return ashape[dim] + bshape[dim];
        else return size;
    }).filter((v: number) => v != 1);
    return a.runKernel(
        "cat",
        { dtype: a.dtype },
        params,
        [output_shape],
        b
    )[0];
}

export function repeat(input: Tensor, shape: Shape): Tensor {
    let t0 = input
    for(let x = 1; x < shape[0]; x++) {
        t0 = cat(t0, input, 0);
    }
    let t1 = t0
    for(let y = 1; shape[1] && y < shape[1]; y++) {
        t1 = cat(t1, t0, 1);
    }
    let t2 = t1;
    for(let z = 1; shape[2] && z < shape[2]; z++) {
        t2 = cat(t2, t1, 2);
    }
    let t3 = t2;
    for(let w = 1; shape[3] && w < shape[3]; w++) {
        t3 = cat(t3, t2, 3);
    }
    return t3;
}

export function layernorm(input: Tensor, normalized_shape: Shape, eps=0.00001): Tensor {
    const params = {
        eps: eps,
        norm_size: normalized_shape.reduce((acc, v) => {
            return acc * v;
        }, 1),
        gamma: 1,
        beta: 0,
        outputSize: input.size

    };
    return input.runKernel(
        "layernorm",
        { dtype: input.dtype },
        params,
        [input.shape]
    )[0];
}

export function maxpool2d(input: Tensor, kernel_size: [number, number], stride: [number, number], padding: [number, number], dilation: [number, number], ceil_mode= false): Tensor {
    if(input.shape.length < 2) throw new Error("MaxPool2d requires a shape that's at least 2d");
    const shape_len = input.shape.length;
    const output_shape = ceil_mode ? 
        [ ...(shape_len > 3 ? [input.shape[shape_len-4]] : []), ...(shape_len > 2 ? [input.shape[shape_len-3]] : []),
            Math.ceil((input.shape[shape_len-2] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1,
            Math.ceil((input.shape[shape_len-1] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
        ] :
        [ ...(shape_len > 3 ? [input.shape[shape_len-4]] : []), ...(shape_len > 2 ? [input.shape[shape_len-3]] : []),
            Math.floor((input.shape[shape_len-2] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1,
            Math.floor((input.shape[shape_len-1] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
        ];
    if(padding[0] != 0) {
        let zero_shape = Array.from(input.shape);
        zero_shape[shape_len-1] = padding[0];
        input = cat(cat(scalar_mul(ones(zero_shape), -256), input, 3), scalar_mul(ones(zero_shape), -256), 3);
    }
    if(padding[1] != 0) {
        let zero_shape = Array.from(input.shape);
        zero_shape[shape_len-2] = padding[1];
        input = cat(cat(scalar_mul(ones(zero_shape), -256), input, 2), scalar_mul(ones(zero_shape), -256), 2);
    }
    const params = {
        kernel_size_x: kernel_size[0],
        kernel_size_y: kernel_size[1],
        stride_x: stride[0],
        stride_y: stride[1],
        padding_x: padding[0],
        padding_y: padding[1],
        dilation_x: dilation[0],
        dilation_y: dilation[1],
        width_y: input.shape[shape_len-1],
        row_len: output_shape[shape_len-1],
        col_len: output_shape[shape_len-2],
        channel_width: shape_len > 2 ? input.shape[shape_len-3] : 1,
        channel_height: shape_len > 3 ? input.shape[shape_len-4] : 1,
        channel_size: input.shape[shape_len-1] * input.shape[shape_len-2],
        outputSize: Tensor.getSize(output_shape)
    };
    return input.runKernel(
        "maxpool2d",
        { dtype: input.dtype },
        params,
        [output_shape]
    )[0];
}
// ------------------------------------
// End Custom
// ------------------------------------

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 * 
 * #### Forward
 * ```
 * output[y, x] = sum(Ky, sum(Kx, input[y + ky, x + kx] * weight[ky, kx])) + bias
 * ```
 * 
 * @param input input tensor of shape [B, inChannels, iH, iW]
 * @param weight filters of shape [outChannels, inChannels, kH, kW]
 * @param bias optional bias tensor of shape [outChannels]
 * @param stride the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
 * @param padding implicit padding on both sides of the kernel. Can be a single number or a tuple (padH, padW). Default: 0
 *     `padding="valid"` is the same as no padding. `padding="same"` pads the input so the output has the shape as the input.
 *     However, this mode can only be used when `stride` is 1.
 * @returns 
 */
export function conv2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | [number, number], padding?: number | [number, number] | "valid" | "same"): Tensor {
    if (shouldCreateGradient(input, weight)) {
        throw new Error("conv2d gradient not supported yet");
    } else {
        if (input.shape.length !== 4 || weight.shape.length !== 4) {
            throw new Error(
                `Expected image tensor, got ${input.shape} and kernel ${weight.shape}`
            );
        }
        if (input.shape[1] !== weight.shape[1]) {
            throw new Error(
                `Expected number of chennels in input image to match number of channels in kernel, got ${input.shape} and ${weight.shape}`
            );
        }
        const params = {
            batchSize: input.shape[0],
            inputChannels: input.shape[1],
            outputChannels: weight.shape[0],
            inputHeight: input.shape[2],
            inputWidth: input.shape[3],
            kernelHeight: weight.shape[2],
            kernelWidth: weight.shape[3],
            outputHeight: input.shape[2] - weight.shape[2] + 1,
            outputWidth: input.shape[3] - weight.shape[3] + 1,
        };
        return input.runKernel(
            "conv2d",
            { dtype: input.dtype },
            params,
            [[params.batchSize, params.outputChannels, params.outputHeight, params.outputWidth]],
            weight
        )[0];
    }
}

export function mm(input: Tensor, other: Tensor): Tensor {
    if (shouldCreateGradient(input, other)) {
        throw new Error("mm gradient not supported yet");
    } else {
        if (input.shape.length !== 2 || other.shape.length !== 2) {
            throw new Error(
                `Expected 2D tensors, got ${input.shape} and ${other.shape}`
            );
        }
        if (input.shape[1] !== other.shape[0]) {
            throw new Error(
                `Expected tensors inner dimensions to be compatible, got ${input.shape} and ${other.shape}`
            );
        }
        const params = {
            resultRows: input.shape[0],
            resultCols: other.shape[1],
            innerDim: input.shape[1],
            alpha: 1.0,
        };
        return input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            params,
            [[params.resultRows, params.resultCols]],
            other
        )[0];
    }
}

export function t(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
        throw new Error(`Expected 2D tensor, got ${input.shape}`);
    }
    if (shouldCreateGradient(input)) {
        throw new Error("t gradient not supported yet");
        // return TransposeFunction.apply(input, 0, 1);
    } else {
        let newShape = input.shape.slice();
        newShape.reverse();
        let newStrides = input.strides.slice();
        newStrides.reverse();
        return input.withShape(newShape, newStrides);
    }
}


export function tensor(spec: TensorSpec): Tensor;
export function tensor(
    array: TensorData,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor;
export function tensor(
    arrayOrSpec: TensorData | TensorSpec,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor {
    return new Tensor(arrayOrSpec, dtype, device, requiresGrad);
}
