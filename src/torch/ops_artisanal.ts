import type { Deviceish } from "./device";
import type { Dtype } from "./dtype";
import { Tensor } from "./tensor";
import { shouldCreateGradient } from "./autograd";
import type { TensorData, TensorSpec } from "./tensor";
import { Shape, defaultStrides, shapeSize , shapesEq} from "./shape";
import * as factories from "./factories";
//import * as ops from "./ops_opgen";

// ------------------------------------
// Start Custom
// ------------------------------------
export function silu(input: Tensor): Tensor {
    return input.runKernel(
        "silu",
        { dtype: input.dtype },
        {
            outputSize: input.size
        },
        [input.shape]
    )[0]
}

export function gelu(input: Tensor): Tensor {
    return input.runKernel(
        "gelu",
        { dtype: input.dtype },
        {
            outputSize: input.size
        },
        [input.shape]
    )[0]
}

export function sin(input: Tensor): Tensor {
    return input.runKernel(
        "sin",
        { dtype: input.dtype },
        {
            outputSize: input.size
        },
        [input.shape]
    )[0]
}

export function cos(input: Tensor): Tensor {
    return input.runKernel(
        "cos",
        { dtype: input.dtype },
        {
            outputSize: input.size
        },
        [input.shape]
    )[0]
}

export function scalar_add(input: Tensor, alpha: number): Tensor {
    return input.runKernel(
        "sadd", { dtype: input.dtype }, {
            alpha: alpha,
            outputSize: input.size
        }, [input.shape]
    )[0]
}

export function scalar_sub(input: Tensor, alpha: number): Tensor {
    return input.runKernel(
        "ssub", { dtype: input.dtype }, {
            alpha: alpha,
            outputSize: input.size
        }, [input.shape]
    )[0]
}

export function scalar_mul(input: Tensor, alpha: number): Tensor {
    return input.runKernel(
        "smul", { dtype: input.dtype }, {
            alpha: alpha,
            outputSize: input.size
        }, [input.shape]
    )[0]
}

export function scalar_div(input: Tensor, alpha: number): Tensor {
    return input.runKernel(
        "sdiv", { dtype: input.dtype }, {
            alpha: alpha,
            outputSize: input.size
        }, [input.shape]
    )[0]
}

export function min(input: Tensor, alpha: number): Tensor {
    return input.runKernel(
        "min", { dtype: input.dtype }, {
            alpha: alpha,
            outputSize: input.size
        }, [input.shape]
    )[0]
}

export function max(input: Tensor, alpha: number): Tensor {
    return input.runKernel(
        "max", { dtype: input.dtype }, {
            alpha: alpha,
            outputSize: input.size
        }, [input.shape]
    )[0]
}



export function find_index(input: Tensor): Tensor {
    return input.runKernel(
        "find_index",
        { dtype: input.dtype },
        {
            outputSize: input.size
        },
        [input.shape]
    )[0]
}

export function clamp(input: Tensor, low: number, high: number): Tensor {
    return input.runKernel(
        "clamp",
        { dtype: input.dtype },
        { 
            low: low,
            high: high,
            outputSize: shapeSize(input.shape) 
        },
        [input.shape],
    )[0]
}

export function cumprod(input: Tensor, dim=0): Tensor {
    return input.runKernel(
        "cumprod",
        { dtype: input.dtype },
        { 
            batchSize: shapeSize(Array.from(input.shape).splice(dim)),
            outputSize: shapeSize(input.shape) 
        },
        [input.shape]
    )[0]
}

export function pow(a: Tensor, b: Tensor): Tensor {
    if(!shapesEq(a.shape, b.shape)) throw new Error(`pow reqires tensors to have the same shape, instead got: ${a.shape} and ${b.shape}`);
    return a.runKernel(
        "pow",
        { dtype: a.dtype },
        { outputSize: shapeSize(a.shape) },
        [a.shape],
        b
    )[0]
}

export function scalar_pow(a: Tensor, alpha: number): Tensor {
    return a.runKernel(
        "scalar_pow",
        { dtype: a.dtype },
        { alpha: alpha, outputSize :shapeSize(a.shape) },
        [a.shape],
    )[0]
}

export function index(input: Tensor, index: Tensor): Tensor {
    /*
    if(input.shape.map((acc, v, i) => {
        return acc || v != index.shape[i];
    }, false)) throw new Error(`index expectes input and index tensors to be the same shape, instead got input shape: ${input.shape} and index shape: ${index.shape}`);
    */
    if(input.shape.length > 1) console.error("index hasn't been built of dimensions greater than 1");
    return input.runKernel(
        "index",
        { dtype: input.dtype },
        { size: index.size },
        [index.shape],
        index
    )[0]
}

export function add(a: Tensor, b: Tensor): Tensor {
    if(shapeSize(a.shape) != shapeSize(b.shape)) throw new Error("trying to add mismatched shapes");
    let ascale = 1;
    let bscale = 1;
    if(a.shape.length < b.shape.length) {
        a.shape.forEach((dim, i) => {
            if(dim != b.shape[i]) throw new Error(`Expected shared dimensions to match, instead got: ${a.shape}, and ${b.shape}`);
        })
        bscale = shapeSize(Array.from(a.shape).splice(a.shape.length));
    } else if(a.shape.length > b.shape.length) {
        b.shape.forEach((dim, i) => {
            if(dim != a.shape[i]) throw new Error(`Expected shared dimensions to match, instead got: ${a.shape}, and ${b.shape}`);
        })
        ascale = shapeSize(Array.from(b.shape).splice(b.shape.length));
    }
    
    return a.runKernel(
        "add",
        { dtype: a.dtype },
        { 
            ascale: ascale,
            bscale: bscale,
            outputSize: shapeSize(a.shape) 
        },
        [a.shape],
        b
    )[0]
}
export function sub(a: Tensor, b: Tensor): Tensor {
    return a.runKernel(
        "sub",
        { dtype: a.dtype },
        { outputSize: shapeSize(a.shape) },
        [a.shape],
        b
    )[0]
}
export function mul(a: Tensor, b: Tensor): Tensor {
    return a.runKernel(
        "mul",
        { dtype: a.dtype },
        { outputSize: shapeSize(a.shape) },
        [a.shape],
        b
    )[0]
}
export function div(a: Tensor, b: Tensor): Tensor {
    //if(shapeSize(a.shape) != shapeSize(b.shape)) throw new Error("trying to add mismatched shapes");

    let ascale = 1;
    let bscale = 1;
    if(a.shape.length < b.shape.length) {
        a.shape.forEach((dim, i) => {
            if(dim != b.shape[i]) throw new Error(`Expected shared dimensions to match, instead got: ${a.shape}, and ${b.shape}`);
        })
        ascale = shapeSize(Array.from(b.shape).splice(a.shape.length));
    } else if(a.shape.length > b.shape.length) {
        b.shape.forEach((dim, i) => {
            if(dim != a.shape[i]) throw new Error(`Expected shared dimensions to match, instead got: ${a.shape}, and ${b.shape}`);
        })
        bscale = shapeSize(Array.from(a.shape).splice(b.shape.length));
    }

    return a.runKernel(
        "div",
        { dtype: a.dtype },
        { 
            ascale: ascale,
            bscale: bscale,
            outputSize: shapeSize(a.shape) 
        },
        [a.shape],
        b
    )[0]
}
export function sqrt(input: Tensor): Tensor {
    return input.runKernel(
        "sqrt",
        { dtype: input.dtype },
        { outputSize: shapeSize(input.shape) },
        [input.shape],
    )[0]
}

export function cat(a: Tensor, b: Tensor, dim: 0|1|2|3): Tensor {
    //throw new Error("cat not implemented yet");
    let part: number;
    let stride: number;
    const shape_len = a.shape.length;
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
    }).filter((v, i) => { return i < shape_len || v != 1});

    return a.runKernel(
        "cat",
        { dtype: a.dtype },
        params,
        [output_shape],
        b
    )[0];
}

export function _repeat(input: Tensor, shape: Shape): Tensor {
    if(input.shape.length > 4) throw new Error(`repeat only supports shapes four dimensions or less, instead got input shape: ${input.shape}`);
    if(shape.length > 4) throw new Error(`repeat only supports shapes four dimensions or less, instead got repeat shape: ${shape}`);

    let t0 = input.copy();
    for(let x = 1; x < shape[0]; x++) {
        t0 = t0.cat(input, 0);
    }
    let t1 = t0.copy();
    for(let y = 1; shape[1] && y < shape[1]; y++) {
        t1 = t1.cat(t0, 1);
    }
    t0.destroy();
    let t2 = t1.copy();
    for(let z = 1; shape[2] && z < shape[2]; z++) {
        t2 = t2.cat(t1, 2);
    }
    t1.destroy();
    let t3 = t2.copy();
    for(let w = 1; shape[3] && w < shape[3]; w++) {
        t3 = t3.cat(t2, 3);
    }
    t2.destroy();
    return t3;
}

export function repeat(input: Tensor, shape: Shape): Tensor {
    if(input.shape.length > 4) throw new Error(`repeat only supports shapes four dimensions or less, instead got input shape: ${input.shape}`);
    if(shape.length > 4) throw new Error(`repeat only supports shapes four dimensions or less, instead got repeat shape: ${shape}`);
    if(input.shape.length != shape.length) throw new Error(`repeat shape must have the same dimention as input shape, instead got input shape: ${input.shape} and repeat shape: ${shape}`);

    const output_shape = input.shape.map((v, i) => { return v * shape[i]; });

    const params = {
        batch_in: input.shape[0],
        batch_repeat: shape[0],
        channel_in: input.dim > 1 ? input.shape[1] : 1,
        channel_repeat: input.dim > 1 ? shape[1] : 1,
        spacial_in: (input.dim > 2 ? input.shape[2]: 1) * (input.dim > 3 ? input.shape[3] : 1),
        spacial_repeat: (input.dim > 2 ? shape[2]: 1) * (input.dim > 3 ? shape[3] : 1),
        outputSize: shapeSize(output_shape)
    }

    return input.runKernel(
        "repeat",
        { dtype: input.dtype },
        params,
        [output_shape]
    )[0]
}

export function layernorm(input: Tensor, normalized_shape: Shape, weight?: Tensor, bias?: Tensor, eps=1e-5): Tensor {
    const params = {
        eps: eps,
        norm_size: shapeSize(normalized_shape),
        outputSize: input.size
    };

    
    if(Array.from(normalized_shape).reduce((acc, v, i) => {
        return acc || v != input.shape[i + (input.shape.length - normalized_shape.length)];
    }, false)) throw new Error(`Layer norm "normalized_shape" must match the 1-n dimensions of the input, instead got input shape: ${input.shape} and normalized_shape: ${normalized_shape}`);

    let needtofreeWeight = false;
    if(typeof(weight) == 'undefined') {
        weight = factories.ones(normalized_shape);
        needtofreeWeight = true;
    }
    let needtofreeBias = false;
    if(typeof(bias) == 'undefined') {
        bias = factories.zeros(normalized_shape);
        needtofreeBias = true;
    }

    const output = input.runKernel(
        "layernorm",
        { dtype: input.dtype },
        params,
        [input.shape],
        weight,
        bias
    )[0];

    if(needtofreeWeight) weight.destroy();
    if(needtofreeBias) bias.destroy();

    return output;
}

export function maxpool2d(input: Tensor, kernel_size: [number, number], stride?:[number,number], padding=[0,0], dilation=[1,1], ceil_mode=false): Tensor {
    if(input.shape.length < 2) throw new Error("MaxPool2d requires a shape that's at least 2d");
    if(typeof(stride) == 'undefined') stride = kernel_size;

    const shape_len = input.shape.length;

    const [h_out, w_out] = ceil_mode ? 
        [
            Math.ceil((input.shape[shape_len-2] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1,
            Math.ceil((input.shape[shape_len-1] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
        ] :
        [ 
            Math.floor((input.shape[shape_len-2] + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1,
            Math.floor((input.shape[shape_len-1] + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1
        ];
    if(padding[0] != 0) {
        let zero_shape = Array.from(input.shape);
        zero_shape[shape_len-1] = padding[0];
        const pad = factories.constant(zero_shape, -256)
        input = cat(cat(pad, input, 3), pad, 3);
        pad.destroy();
    }
    if(padding[1] != 0) {
        let zero_shape = Array.from(input.shape);
        zero_shape[shape_len-2] = padding[1];
        const pad = factories.constant(zero_shape, -256)
        input = cat(cat(pad, input, 2), pad, 2);
        pad.destroy();
    }

    const output_shape = [...Array.from(input.shape).splice(0, input.shape.length-2), h_out, w_out]

    const params = {
        batches: shapeSize(output_shape) / h_out / w_out,
        output_height: h_out,
        output_width: w_out,
        input_height: input.shape[input.shape.length-2],
        input_width: input.shape[input.shape.length-1], 
        kernel_size_x: kernel_size[0],
        kernel_size_y: kernel_size[1],
        stride_x: stride[0],
        stride_y: stride[1],
        outputSize: shapeSize(output_shape),
    };

    return input.runKernel(
        "maxpool2d",
        { dtype: input.dtype },
        params,
        [output_shape]
    )[0];
}

export function upsample(
    input: Tensor, 
    size: number | [number, number] | [number, number, number] | null = null,
    scale_factor: number | [number, number] | [number, number, number] | null = null,
    mode: "nearest" | "linear" | "bilinear" | "bicubic" | "trilinear" = "nearest",
    align_corners=false,
    recompute_scale_factor=false,
) {
    if(align_corners) throw new Error("align_corners not implemented!");
    if(recompute_scale_factor) throw new Error("recompute_scale_factor not implemented!");
    if(mode == "trilinear") throw new Error("trilinear not implemented!");
    let output_shape;
    if(size == null) {
        if(scale_factor == null) throw new Error("both size and scale factor cannot be undefined at once");
        if(typeof(scale_factor) == 'number') scale_factor = (new Array(input.shape.length - 2)).fill(scale_factor) as any
        else if(scale_factor.length != input.shape.length - 2) throw new Error(`Expects a ${input.shape.length - 2}D scale factor for a ${input.shape.length}D input`);
        output_shape = input.shape.map((v, i) => {
            if(i < 2) return v;
            return Math.floor(v * scale_factor[i-2]);
        })
    } else {
        if(scale_factor != null) throw new Error("both size and scale factor cannot be defined at once");
        if(typeof(size) == 'number') size = (new Array(input.shape.length - 2)).fill(size) as any
        else if(size.length != input.shape.length - 2) throw new Error(`Expects a ${input.shape.length - 2}D scale factor for a ${input.shape.length}D input`);
        output_shape = input.shape.map((v, i) => {
            if(i < 2) return v;
            return size[i-2]
        })
    }
    if(input.shape.length == 3 && !(mode == "nearest" || mode == "linear")) throw new Error("The only acceptable modes for a 3D tensor are \"nearest\" and \"linear\"");
    else if(input.shape.length == 4 && !(mode == "nearest" || mode == "bilinear" || mode == "bicubic")) throw new Error("The only acceptable modes for a 4D tensor are \"bilinear\" and \"bicubic\"");
    else if(input.shape.length == 5 && !(mode == "nearest" /* ||  mode == "trilinear" */)) throw new Error("The only acceptable mode for a 5D tensor is \"trilinear\"");
    
    const params = {
        mode: ["nearest", "linear", "bilinear", "bicubic", "trilinear"].indexOf(mode),
        w_in: input.shape[2],
        h_in: input.shape[3] ? input.shape[3] : 1,
        d_in: input.shape[4] ? input.shape[4] : 1,
        w_out: output_shape[2],
        h_out: output_shape[3] ? output_shape[3] : 1,
        d_out: output_shape[4] ? output_shape[4] : 1,
        outputSize: shapeSize(output_shape)
    }
    return input.runKernel(
        "upsample",
        { dtype: input.dtype},
        params,
        [output_shape]
    )[0]
}

export function rand(
    input: Tensor,
    seed: number,
): { output: Tensor, next_seed: number } {

    const params = {
        seed: seed,
        outputSize: input.size
    }
    return {
        output: input.runKernel(
            "rand",
            { dtype: input.dtype },
            params,
            [input.shape]
        )[0],
        next_seed: seed + params.outputSize
    }
}

export function box_muller(
    input: Tensor,
    mean: number,
    std: number
): Tensor {
    const params = {
        mean: mean,
        sdev: std,
        outputSize: shapeSize(input.shape) / 2
    }
    const output_shape = input.shape.splice(1);
    return input.runKernel(
        "box_muller",
        { dtype: input.dtype },
        params,
        [output_shape],
    )[0];
}

export function clt(
    input: Tensor,
    sample_size: number,
    mean: number,
    std: number
): Tensor {
    const output_shape = input.shape.splice(0,input.shape.length-1);
    const params = {
        sample_size: sample_size,
        mean: mean,
        sdev: std,
        outputSize: shapeSize(output_shape)
    }
    return input.runKernel(
        "clt",
        { dtype: input.dtype },
        params,
        [output_shape],
    )[0];
}

export function squeeze(
    input: Tensor,
    dim?: number
): Tensor {
    if(dim <= -input.dim || dim > input.dim) throw new Error(`"dim" out of bounds; must be on the interval:  [${-input.dim}, ${input.dim}) for input tensor with dim: ${input.dim}`);
    if(dim < 0) dim = dim + input.dim;
    let new_shape = input.shape;
    if(dim) {
        if(dim < 0) dim = input.shape.length + dim;
        if(input.shape[dim] == 1) new_shape = [...input.shape.slice(0,dim), ...input.shape.slice(dim + 1)];
    } else {
        new_shape = input.shape.filter((size: number) => { return size != 1 });
    }
    return input.withShape(new_shape, defaultStrides(new_shape));
}

export function unsqueeze(
    input: Tensor,
    dim: number
): Tensor {
    if(dim <= -input.dim - 1 || dim > input.dim + 1) throw new Error(`"dim" out of bounds; must be on the interval:  [${-input.dim - 1}, ${input.dim + 1}) for input tensor with dim: ${input.dim}`);
    if(dim < 0) dim = dim + input.dim + 1;
    const new_shape = [...input.shape.slice(0, dim), 1, ...input.shape.slice(dim)];
    return input.withShape(new_shape, defaultStrides(new_shape));
}

export function linear(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
): Tensor {
    const output_shape = Array.from(input.shape);
    const feature_dim = input.shape.length - 1;

    if(typeof(bias) == 'undefined') {
        bias = factories.zeros(weight.shape[0]);
    } else {
        bias = bias.copy();
    }


    if(weight.shape[1] != input.shape[feature_dim]) throw new Error(`Expected last dimention of input and last dimention of weight to match, instead got input.shape = ${input.shape} & weight.shape = ${weight.shape}`);
    if(bias && bias.shape[0] != weight.shape[0]) throw new Error(`Expected bias to be 1D and match the first dimention of weight, instead got bias.shape = ${bias.shape} and weight.shape = ${weight.shape}`);
    
    output_shape[feature_dim] = weight.shape[0];
    let output = mm(input.view([-1, input.shape[feature_dim]]), transpose(weight, 0, 1));
    output = output.add(repeat(bias.unsqueeze(0), [output.shape[0], 1]));
    output = output.view(output_shape)

    bias.destroy();
    return output;
}

export function sum(
    input: Tensor,
    dim=0
): Tensor {
    if(dim < 0 || dim >= input.shape.length) throw new Error(`Expected dim that's within the tensor but input has shape: ${input.shape} and got dim ${dim}`);
    const output_shape = dim == 0 ? [1] : Array.from(input.shape).splice(0, dim);

    let output = input;
    let batch_size = shapeSize(Array.from(output.shape).splice(dim));
    while(batch_size % 2 == 0) {
        const pass_shape = [...output_shape, batch_size / 2];
        let batches = shapeSize(pass_shape);
        output = output.runKernel(
            "sum", 
            { dtype: input.dtype }, 
            { 
                batches: batches,
                batch_size: 2,
                size: input.size 
            }, 
            [pass_shape]
        )[0];
        batch_size = shapeSize(Array.from(output.shape).splice(dim));
    }

    return output.runKernel(
        "sum", 
        { dtype: input.dtype }, 
        { 
            batches: shapeSize(output_shape),
            batch_size: shapeSize(Array.from(output.shape).splice(dim)),
            size: output.size 
        }, 
        [output_shape]
    )[0];
}

export function exp(
    input: Tensor
): Tensor {
    return input.runKernel(
        "exp",
        { dtype: input.dtype },
        { outputSize: shapeSize(input.shape) },
        [input.shape]
    )[0]
}

export function softmax(
    input: Tensor,
    dim=0
): Tensor {
    if(dim < 0 || dim >= input.shape.length) throw new Error(`Expected dim that's within the tensor but input has shape: ${input.shape} and got dim ${dim}`);

    const batches = dim == 0 ? 1 : shapeSize(Array.from(input.shape).splice(0, dim));
    const batch_size = input.shape[dim];
    const stride = dim+1 == input.shape.length ? 1 : shapeSize(Array.from(input.shape).splice(dim+1));

    const params = {
        batches: batches,
        batch_size: batch_size,
        stride: stride
    };

    const exps = exp(input).transpose(dim, input.shape.length-1);
    const sums = sum(exps, input.shape.length-1);
    const output = div(exps, sums).transpose(dim, input.shape.length-1);
    exps.destroy();
    sums.destroy();
    return output;
}

export function permute(
    input: Tensor,
    dims: Array<number>,
): Tensor {
    if(dims.length != input.shape.length) throw new Error(`permute need the same number of dimentions as it's input, got: ${dims} but expected: ${input.shape} dimensions`);
    const seen = [];
    dims.forEach((v) => {
        if(seen.includes(v)) throw new Error(`Permute cannot take duplicated dims: ${dims}`);
        seen.push(v);
    });

    input = input.copy();
    
    for(let i = 0; i < dims.length; i++) {
        if(dims[i] < 0 || dims[i] >= dims.length) throw new Error(`No dimention ${dims[i]} in input shape: ${input.shape}`);

        if(i != dims[i]) {
            input = input.transpose(i, dims[i]);
            dims[dims.indexOf(i)] = dims[i];
        }
        
    }
    
    return input
}

export function scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask?: Tensor,
    dropout?: number,
    is_casual?: boolean,
): Tensor {
    if(attn_mask) console.error("attention does not support attention masks");
    if(dropout) console.error("attention does not support dropout");
    if(is_casual) console.error("attention does not support is_causal");

    const output_shape = Array.from(query.shape);
    output_shape[output_shape.length-1] = value.shape[value.shape.length-1];

    const sqrt_dk = Math.sqrt(key.shape[key.shape.length-1]);

    query = query.view([-1, query.shape[query.shape.length-2], query.shape[query.shape.length-1]]);
    key = key.view([-1, key.shape[key.shape.length-2], key.shape[key.shape.length-1]]);
    value = value.view([-1, value.shape[value.shape.length-2], value.shape[value.shape.length-1]]);

    /*
    let out = scalar_div(mm(query, key.transpose(1,2)), sqrt_dk);
    out = softmax(out, 1);
    out = mm(out, value);
    return out.view(output_shape);
    */

    // batch the attention calculations
    const batches = query.shape[0];
    const query_batches = chunk(query, batches, 0);
    const key_batches = transpose(key, 1,2).chunk(batches, 0);
    const value_batches = chunk(value, batches);

    //console.log("dot product input shapes: ", query_batches[0].shape, key_batches[0].shape);
    const dot_products = query_batches.map((q, i) => {
        const out = q.mm(key_batches[i]).scalar_div(sqrt_dk);
        key_batches[i].destroy();
        return out;
    });
    //console.log("dot product shape: ", dot_products[0].shape);

    const softmaxxes = dot_products.map((dot_product) => {
        return dot_product.softmax(dot_product.shape.length-1);
    });
    //console.log("softmaxxes shape: ", softmaxxes[0].shape);

    
    let outs = softmaxxes.map((batch, i) => {
        const out = batch.mm(value_batches[i]);
        value_batches[i].destroy();
        return out;
    })
    //console.log("outs shape: ", outs[0].shape);
    
    let out = outs[0];
    for(let i = 1; i < batches; i++) {
        out = out.cat(outs[i], 0);
        outs[i].destroy();
    }
    return out.view(output_shape);
}


export function multihead_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: number,
    num_heads: number,

    in_proj_weight: Tensor | null,
    in_proj_bias: Tensor | null,
    
    add_zero_attn: boolean,
    dropout_p: number,
    out_proj_weight: Tensor,
    out_proj_bias?: Tensor,

    key_padding_mask?: Tensor,
    attn_mask?: Tensor,

    use_separate_proj_weight=false,
    q_proj_weight?: Tensor,
    k_proj_weight?: Tensor,
    v_proj_weight?: Tensor,
    static_k?: Tensor,
    static_v?: Tensor,
    need_weights=false,
    is_causal=false
): { output: Tensor, weights: Tensor | null } {

    const is_batched = _mha_shape_check(query, key, value, num_heads, key_padding_mask, attn_mask);

    if(!is_batched) {
        query = query.unsqueeze(1);
        key = key.unsqueeze(1);
        value = value.unsqueeze(1);
        if(key_padding_mask != null) key_padding_mask = key_padding_mask.unsqueeze(0);
    }


    const tgt_len = query.shape[0];
    const bsz = query.shape[1];
    const embed_dim = query.shape[2];
    const src_len = key.shape[0];

    
    if(embed_dim != embed_dim_to_check) throw new Error(`was expecting embedding dimension of ${embed_dim_to_check}, but got ${embed_dim}`);
    const head_dim = Math.floor(embed_dim / num_heads);

    let q, k, v;
    if(!use_separate_proj_weight) {
        if(in_proj_weight == null) throw new Error("use_separate_proj_weight is False but in_proj_weight is null")
        const unpacked_weights = chunk(in_proj_weight, 3);
        q_proj_weight = unpacked_weights[0];
        k_proj_weight = unpacked_weights[1];
        v_proj_weight = unpacked_weights[2];
    } else {
        if(q_proj_weight == null) throw new Error("use_separate_proj_weight is True but q_proj_weight is null");
        if(k_proj_weight == null) throw new Error("use_separate_proj_weight is True but k_proj_weight is null");
        if(v_proj_weight == null) throw new Error("use_separate_proj_weight is True but v_proj_weight is null");
    }

    let b_q, b_k, b_v;
    if(in_proj_bias == null) {
        b_q = null;
        b_k = null;
        b_v = null;
    } else {
        const chunks = chunk(in_proj_bias, 3);
        b_q = chunks[0];
        b_k = chunks[1];
        b_v = chunks[2];
    }

    const proj = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v);
    
    q_proj_weight.destroy();
    k_proj_weight.destroy();
    v_proj_weight.destroy();
    b_q.destroy();
    b_k.destroy();
    b_v.destroy();

    q = proj.q;
    k = proj.k;
    v = proj.v;

    q = q.view([tgt_len, bsz * num_heads, head_dim]).transpose(0, 1)
    if (typeof(static_k) == 'undefined') {
        k = k.view([k.shape[0], bsz * num_heads, head_dim]).transpose(0, 1)
    } else {
        if(static_k.shape[0] != bsz*num_heads) throw new Error(`expecting static_k.shape[0] of ${bsz * num_heads}, but got ${static_k.shape[0]}`);
        if(static_k.shape[2] != head_dim) throw new Error(`expecting static_k.shape[0] of ${head_dim}, but got ${static_k.shape[0]}`);
        k = static_k;
    }
    if (typeof(static_v) == 'undefined') {
        v = v.view([v.shape[0], bsz * num_heads, head_dim]).transpose(0, 1)
    } else {
        if(static_v.shape[0] != bsz*num_heads) throw new Error(`expecting static_k.shape[0] of ${bsz * num_heads}, but got ${static_v.shape[0]}`);
        if(static_v.shape[2] != head_dim) throw new Error(`expecting static_k.shape[0] of ${head_dim}, but got ${static_v.shape[0]}`);
        v = static_v;
    }

    if (need_weights) {
        console.error("need weights");
    } else {
        /*
        if(attn_mask != null) {
            if(attn_mask.shape[0] == 1 && attn_mask.dim == 3) {
                attn_mask = attn_mask.unsqueeze(0);
            } else {
                attn_mask = attn_mask.view([bsz, num_heads, -1, src_len]);
            }
        }
        */

        q = q.view([bsz, num_heads, tgt_len, head_dim]);
        k = k.view([bsz, num_heads, src_len, head_dim]);
        v = v.view([bsz, num_heads, src_len, head_dim]);

        // q in missing by here
        let attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal);
        attn_output = attn_output.permute([2, 0, 1, 3]).view([bsz * tgt_len, embed_dim]);

        attn_output = attn_output.linear(out_proj_weight, out_proj_bias);
        attn_output = attn_output.view([tgt_len, bsz, attn_output.shape[1]]);
        if(!is_batched) {
            attn_output = attn_output.squeeze(1);
        }
        return { output: attn_output, weights: null };
    }
}

function _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q?: Tensor,
    b_k?: Tensor,
    b_v?: Tensor,
): { q: Tensor, k: Tensor, v: Tensor } {

    const Eq = q.shape[q.dim-1];
    const Ek = k.shape[k.dim-1];
    const Ev = v.shape[v.dim-1];
    if(w_q.shape[0] != Eq || w_q.shape[1] != Eq) throw new Error(`expecting query weights shape of ${[Eq, Eq]}, but got ${w_q.shape}`);
    if(w_k.shape[0] != Eq || w_k.shape[1] != Ek) throw new Error(`expecting key weights shape of ${[Eq, Ek]}, but got ${w_k.shape}`);
    if(w_v.shape[0] != Eq || w_v.shape[1] != Ev) throw new Error(`expecting value weights shape of ${[Eq, Ev]}, but got ${w_v.shape}`);
    if(b_q && b_q.shape[0] != Eq) throw new Error(`expecting query bias shape of ${[Eq]}, but got ${b_q.shape}`);
    if(b_k && b_k.shape[0] != Eq) throw new Error(`expecting key bias shape of ${[Eq]}, but got ${b_k.shape}`);
    if(b_v && b_v.shape[0] != Eq) throw new Error(`expecting value bias shape of ${[Eq]}, but got ${b_v.shape}`);
    return { 
        q: linear(q, w_q, b_q), 
        k: linear(k, w_k, b_k), 
        v: linear(v, w_v, b_v)
    }
}

export function chunk(
    input: Tensor,
    chunks: number,
    dim=0
): Array<Tensor> {
    if(input.shape[dim] % chunks != 0) throw new Error(`cannot chunk input of shape: ${input.shape} into ${chunks} even chunks along dim: ${dim}`);
    const output_shape = input.shape.map((v: number, i: number) => {
        if(i == dim) return v / chunks;
        return v;
    })

    const params = {
        chunkSize: (dim+1 < output_shape.length ? shapeSize(Array.from(output_shape).splice(dim + 1)) : 1) * output_shape[dim],
        numChunks: chunks,
        outputSize: shapeSize(output_shape)
    }
    const arr = [];
    for(let i = 0; i < chunks; i++) {
        arr.push(input.runKernel(
            "chunk",
            { dtype: input.dtype },
            {...params, offset: i},
            [output_shape] 
        )[0]);
    }
    return arr;
}

function _mha_shape_check(query: Tensor, key: Tensor, value: Tensor, num_heads: number, key_padding_mask?: Tensor, attn_mask?: Tensor) {
    let is_batched;

    if (query.dim == 3) {
        // batched input
        is_batched = true;
        if(key.dim != 3 || value.dim != 3) 
            throw new Error(`For batched (3-D) "query", expected "key" and "value" to be 3-D but found ${key.dim}-D and ${value.dim}-D tensors respectively`);
        if(key_padding_mask != undefined)
            if(key_padding_mask.dim != 2) 
                throw new Error(`For batched (3-D) "query", expected "key_padding_mask" to be "undefined" or 2-D but found ${key_padding_mask.dim}-D tensor instead`);
        if(attn_mask != undefined)
            if(attn_mask.dim != 2 && attn_mask.dim != 3)
                throw new Error(`For batched (3-D) "query", expected "attn_mask" to be "None", 2-D or 3-D but found ${attn_mask.dim}-D tensor instead`);
    } else if(query.dim == 2) {
        // unbatched input
        is_batched = false;
        if(key.dim != 2 || value.dim != 2) 
            throw new Error(`For unbatched (2-D) "query", expected "key" and "value" to be 2-D but found ${key.dim}-D and ${value.dim}-D tensors respectively`);
        if(key_padding_mask != undefined)
            if(key_padding_mask.dim != 1) 
                throw new Error(`For unbatched (2-D) "query", expected "key_padding_mask" to be "undefined" or 1-D but found ${key_padding_mask.dim}-D tensor instead`);
        if(attn_mask != undefined)
            if(attn_mask.dim != 2 && attn_mask.dim != 3)
                throw new Error(`For batched (2-D) "query", expected "attn_mask" to be "None", 2-D or 3-D but found ${attn_mask.dim}-D tensor instead`);
            if(attn_mask.dim == 3) {
                const expected_shape = [num_heads, query.shape[0], key.shape[0]];
                if(expected_shape.length != attn_mask.shape.length || 0 == expected_shape.reduce((acc, v, i) => { return acc != 0 ? 1 : (v == attn_mask.shape[i] ? 0 : 1)}, 0))
                    throw new Error(`Exptected "attn_mask" shape to be ${expected_shape} but got ${attn_mask.shape}`);
            }
    } else {
        throw new Error(`"query" should be unbatched 2D or batched 3D tensor but received ${query.dim}-D query tensor`);
    }
    return is_batched;
}

export function group_norm(input: Tensor, groups: number, weight?: Tensor, bias?: Tensor, eps=1e-5): Tensor {
    if(input.dim < 2) throw new Error("group norm expects at least two dimenstions");
    if(input.shape[1] % groups != 0) throw new Error(`group norm expects group num to evenly divide channels but got input shape: ${input.shape} and group num: ${groups} (${input.shape[1] / groups})`);

    if(weight.shape[0] != input.shape[1] || weight.dim > 1) throw new Error(`in group_norm, weights should be the same size as input channels, instead got input shape: ${input.shape} and weight shape: ${weight.shape}`);
    if(bias.shape[0] != input.shape[1] || bias.dim > 1) throw new Error(`in group_norm, bias should be the same size as input channels, instead got input shape: ${input.shape} and bias shape: ${weight.shape}`);
    

    const params = {
        eps: eps,
        batches: input.shape[0],
        channels: input.shape[1],
        groups: groups,
        groupSize: shapeSize(Array.from(input.shape).splice(1)) / groups,
        outputSize: shapeSize(input.shape)
    }

    let needtofreeWeight = false;
    if(!weight) {
        weight = factories.ones([input.shape[1]]);
        needtofreeWeight = true;
    }
    let needtofreeBias = false;
    if(!bias) {
        bias = factories.zeros([input.shape[1]]);
        needtofreeBias = true;
    }

    const output = input.runKernel(
        "group_norm",
        { dtype: input.dtype },
        params,
        [input.shape],
        weight,
        bias
    )[0];

    if(needtofreeWeight) weight.destroy();
    if(needtofreeBias) bias.destroy();

    return output;
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
export function conv2d(input: Tensor, weight: Tensor, bias?: Tensor, stride: number | [number, number]=1, padding: number | [number, number] | "valid" | "same"=0, dilation: number | [number, number]=1, groups=1): Tensor {
    if (shouldCreateGradient(input, weight)) {
        //throw new Error("conv2d gradient not supported yet");
        //console.error("conv2d gradient not supported yet");
    }
    if (input.shape.length !== 4 || weight.shape.length !== 4) {
        throw new Error(
            `Expected image tensor, got input: [Cout, Cin], got ${input.shape} and kernel ${weight.shape}`
        );
    }
    if (input.shape[1] !== weight.shape[1]) {
        throw new Error(
            `Expected number of channels in input image to match number of channels in kernel, got ${input.shape} and ${weight.shape}`
        );
    }

    let needtofreeBias = false;
    if(!bias) {
        bias = factories.zeros(weight.shape[0]);
        needtofreeBias = true;
    }

    if(weight.shape[1] != input.shape[1] / groups) throw new Error(`Expected weight to match shape [out_channels, in_channels / groups, kH, kW] but got input: ${input.shape} and weight: ${weight.shape}`)

    if(typeof(padding) != 'undefined') {
        if(typeof(padding) == 'number') padding = [padding, padding];
        if(padding[0] != 0) {
            const zero_shape = Array.from(input.shape);
            zero_shape[input.shape.length-1] = padding[0] as any;
            input = cat(cat(factories.zeros(zero_shape), input, 3), factories.zeros(zero_shape), 3);
        }
        if(padding[1] != 0) {
            const zero_shape = Array.from(input.shape);
            zero_shape[input.shape.length-2] = padding[1] as any;
            input = cat(cat(factories.zeros(zero_shape), input, 2), factories.zeros(zero_shape), 2);
        }
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

    const output = input.runKernel(
        "conv2d",
        { dtype: input.dtype },
        params,
        [[params.batchSize, params.outputChannels, params.outputHeight, params.outputWidth]],
        weight,
        bias
    )[0];

    if(needtofreeBias) bias.destroy();
    return output;
}

export function mm(input: Tensor, other: Tensor): Tensor {
    if (shouldCreateGradient(input, other)) {
        throw new Error("mm gradient not supported yet");
    } else {
        let is_batched;
        if(input.shape.length == 2) is_batched = false;
        else if(input.shape.length == 3) is_batched = true;
        else throw new Error(`Expected 2D tensors or 3D tensors (batched) got ${input.shape} and ${other.shape}`)

        if(input.shape.length != other.shape.length) throw new Error(`Expected tensors to be of the same dimension, instead got got ${input.shape} and ${other.shape}`);
        if(is_batched && input.shape[0] != other.shape[0]) throw new Error(`Expected tensors to have the same number of batches, instead got got ${input.shape} and ${other.shape}`);
        if ((!is_batched && (input.shape[1] !== other.shape[0])) || (is_batched && (input.shape[2] !== other.shape[1]))) {
            throw new Error(
                `Expected tensors inner dimensions to be compatible, got ${input.shape} and ${other.shape}`
            );
        }
        
        const params = {
            batches: is_batched ? input.shape[0] : 1,
            resultRows: is_batched ? input.shape[1] : input.shape[0],
            resultCols: is_batched ? other.shape[2] : other.shape[1],
            trueInnerDim: is_batched ? input.shape[2] : input.shape[1],
            alpha: 1.0,
            innerDim: is_batched ? input.shape[2] : input.shape[1],
            innerDimOffset: 0
        };

        //console.log("running linear with params: ", params);
        //console.log("output shape: ", [...(is_batched ? [params.batches] : []), params.resultRows, params.resultCols])
        /*
        const a = input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            { ...params, innerDim: params.trueInnerDim / 4, innerDimOffset: 0 },
            [[...(is_batched ? [params.batches] : []), params.resultRows, params.resultCols]],
            other
        )[0];
        const b = input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            { ...params, innerDim: params.trueInnerDim * 2 / 4, innerDimOffset: params.trueInnerDim / 4 },
            [[...(is_batched ? [params.batches] : []), params.resultRows, params.resultCols]],
            other
        )[0];
        const c = input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            { ...params, innerDim: params.trueInnerDim * 3 / 4, innerDimOffset: params.trueInnerDim * 2 / 4 },
            [[...(is_batched ? [params.batches] : []), params.resultRows, params.resultCols]],
            other
        )[0];
        const d = input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            { ...params, innerDim: params.trueInnerDim, innerDimOffset: params.trueInnerDim * 3 / 4 },
            [[...(is_batched ? [params.batches] : []), params.resultRows, params.resultCols]],
            other
        )[0];
        return add(add(add(a, b), c), d);
        */
        return input.runKernel(
            "mm",
            { resultDtype: input.dtype },
            params,
            [[...(is_batched ? [params.batches] : []), params.resultRows, params.resultCols]],
            other
        )[0];
    }
}

export function transpose(input: Tensor, dim0=0, dim1=1): Tensor {
    if(dim1 == dim0) {
        return input.copy();
    } else if(dim1 < dim0) {
        const temp = dim0;
        dim0 = dim1;
        dim1 = temp;
    }
    if (shouldCreateGradient(input)) {
        //console.error("transpose gradient not supported yet");
        // return TransposeFunction.apply(input, 0, 1);
    }
    const newShape = Array.from(input.shape);
    newShape[dim0] = input.shape[dim1];
    newShape[dim1] = input.shape[dim0];
    const params = {
        dim0: input.shape[dim0],
        batchSize: shapeSize(Array.from(newShape).splice(dim0)),
        dim1: input.shape[dim1],
        elSize: shapeSize(Array.from(newShape).splice(dim1)) / newShape[dim1],
        stride: dim1-dim0 == 1 ? 1 : shapeSize(Array.from(newShape).splice(dim0+1, dim1-dim0-1)),
        outputSize: input.size
    }

    return input.runKernel(
        "transpose", 
        { dtype: input.dtype },
        params,
        [newShape]
    )[0];
}

export function copy(input: Tensor): Tensor {
    return input.runKernel(
        "copy",
        { dtype: input.dtype },
        { outputSize: shapeSize(input.shape)},
        [input.shape]
    )[0]
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
