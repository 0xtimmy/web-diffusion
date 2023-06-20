import type { Deviceish } from "./device";
import type { Dtype } from "./dtype";
import { Tensor } from "./tensor";
import { shouldCreateGradient } from "./autograd";
import type { TensorData, TensorSpec } from "./tensor";
import { Shape, defaultStrides, shapeSize } from "./shape";
import { ones, zeros } from "./factories";
import * as ops from "./ops_opgen";

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

export function layernorm(input: Tensor, normalized_shape: Shape, eps=1e-5): Tensor {
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
        input = cat(cat(ops.scalar_mul(ones(zero_shape), -256), input, 3), ops.scalar_mul(ones(zero_shape), -256), 3);
    }
    if(padding[1] != 0) {
        let zero_shape = Array.from(input.shape);
        zero_shape[shape_len-2] = padding[1];
        input = cat(cat(ops.scalar_mul(ones(zero_shape), -256), input, 2), ops.scalar_mul(ones(zero_shape), -256), 2);
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
        outputSize: shapeSize(output_shape)
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
    size: number | [number, number] | [number, number, number] | null,
    scale_factor: number | [number, number] | [number, number, number] | null ,
    mode: "nearest" | "linear" | "bilinear" | "bicubic" | "trilinear",
    align_corners: boolean,
    recompute_scale_factor: boolean
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
    const output_shape = input.shape;
    const params = {
        mean: mean,
        std: std,
        outputSize: input.size
    }
    return input.runKernel(
        "box_muller",
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
    let output = mm(input.unsqueeze(0), weight.t()).squeeze(0);
    if(bias) return ops.add(output, bias);
    else return output;
}

export function softmax(
    input: Tensor,
    dim?: number,
): Tensor {

    const exp = input.exp();
    const sum_exp = exp.sum(dim);

    const params = {
        stride: shapeSize(input.shape.slice(dim)),
        outputSize: input.size
    };
    return exp.runKernel(
        "softmax",
        { dtype: input.dtype },
        params,
        [input.shape],
        sum_exp
    )[0]
}

export function permute(
    input: Tensor,
    dims: Array<number>,
): Tensor {
    if(dims.length != input.shape.length) throw new Error("permute need the same number of dimentions as it's input");
    const seen = [];
    dims.forEach((v) => {
        if(seen.includes(v)) throw new Error(`Permute cannot take duplicate dimentions (${dims})`);
        if(v < 0 || v >= dims.length) throw new Error(`No dimention ${v} in input shape: ${input.shape}`);
    })
    const output_shape = dims.map((v: number) => {
        return input.shape[v];
    });
    return input.view(output_shape);
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
    return mm(softmax(ops.scalar_div(mm(query, key.t()), Math.sqrt(key.size))), value);
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
    const embed_dim = query.shape[3];
    const src_len = key.shape[0];

    
    if(embed_dim != embed_dim_to_check) throw new Error(`was expeecting embedding dimension of ${embed_dim_to_check}, but got ${embed_dim}`);
    const head_dim = embed_dim;

    let q, k, v;
    if(!use_separate_proj_weight) {
        if(in_proj_weight == null) throw new Error("use_separate_proj_weight is False but in_proj_weight is null")

        const unpacked_weights = in_proj_weight.chunk(3);
        q_proj_weight = unpacked_weights[0];
        k_proj_weight = unpacked_weights[1];
        v_proj_weight = unpacked_weights[2];
    }
    if(q_proj_weight == null) throw new Error("use_separate_proj_weight is False but q_proj_weight is null");
    if(k_proj_weight == null) throw new Error("use_separate_proj_weight is False but k_proj_weight is null");
    if(v_proj_weight == null) throw new Error("use_separate_proj_weight is False but v_proj_weight is null");
    let b_q, b_k, b_v;
    if(in_proj_bias == null) {
        b_q = null;
        b_k = null;
        b_v = null;
    } else {
        const chunks = in_proj_bias.chunk(3);
        b_q = chunks[0];
        b_k = chunks[1];
        b_v = chunks[2]
    }
    const proj = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
    q = proj.q;
    k = proj.k;
    v = proj.v;


    if (need_weights) {
        console.error("need weights");
    } else {
        if(attn_mask != null) {
            if(attn_mask.shape[0] == 1 && attn_mask.dim == 3) {
                attn_mask = attn_mask.unsqueeze(0);
            } else {
                attn_mask = attn_mask.view([bsz, num_heads, -1, src_len]);
            }
        }

        q = q.view(bsz, num_heads, tgt_len, head_dim);
        k = k.view(bsz, num_heads, src_len, head_dim);
        v = v.view(bsz, num_heads, src_len, head_dim);

        let attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal);
        attn_output = attn_output.permute([2, 0, 1, 3]).view([bsz * tgt_len, embed_dim]);

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias);
        attn_output = attn_output.view([tgt_len, bsz, attn_output.shape[1]]);
        if(!is_batched) {
            attn_output = attn_output.squeeze(1);
        }
        return { output: attn_output, weights: null };
    }
    
   return { output: query, weights: null };
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
    return { q: linear(q, w_q, b_q), k: linear(k, w_k, b_k), v: linear(v, w_v, b_v)}
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
    const output_shapes: Array<Shape> = (new Array(chunks)).map(() => output_shape);

    const params = {
        chunkSize: output_shape.reduce((acc: number, v: number, i: number) => { return i > dim ? acc * v : acc; }, 1),
        stride: chunks,
        strides: output_shape.reduce((acc: number, v: number, i: number) => { return i < dim ? acc * v : acc; }, 1),
    }

    return (new Array(chunks)).map((_, i: number): Tensor => {
        return input.runKernel(
            "chunk",
            { dtype: input.dtype },
            {...params, offset: i},
            output_shapes 
        )[0]
    });
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

    const params = {
        eps: eps,
        groupSize: input.shape[1] / groups,
        outputSize: shapeSize(input.shape)
    }

    if(!weight) weight = ones([input.shape[0], groups]);
    if(!bias) bias = zeros([input.shape[0], groups]);

    return input.runKernel(
        "group_norm",
        { dtype: input.dtype },
        params,
        [input.shape],
        weight,
        bias
    )[0]
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
export function conv2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | [number, number], padding?: number | [number, number] | "valid" | "same", dilation?: number | [number, number], groups?: number): Tensor {
   
    if (shouldCreateGradient(input, weight)) {
        //throw new Error("conv2d gradient not supported yet");
        console.error("conv2d gradient not supported yet");
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

export function transpose(input: Tensor, dim0=0, dim1=1): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("t gradient not supported yet");
        // return TransposeFunction.apply(input, 0, 1);
    } else {
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
        console.log("running transpose with params: ", params);
        return input.runKernel(
            "transpose", 
            { dtype: input.dtype },
            params,
            [newShape]
        )[0];
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
