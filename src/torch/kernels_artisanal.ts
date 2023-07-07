import { KernelSpec, KernelConfigSpec, KernelParamSpec, KernelInputSpec, KernelOutputSpec } from "./kernel";
import { ExprCode } from "./expr";

function _defaultKernel({
    name,
    config =  [{ name: "dtype" }],
    parameters = [{ name: "outputSize", shaderType: "u32" }],
    inputs = [{ name: "input", shaderType: "array<f32>"}],
    outputs = [{ name: "output", shaderType: "array<f32>", size: "outputSize" }],
    workgroupCount = ["parameters.outputSize", 1, 1],
    shader
}: {
    name: string,
    config?: Array<KernelConfigSpec>,
    parameters?: Array<KernelParamSpec>,
    inputs?: Array<KernelInputSpec>,
    outputs?: Array<KernelOutputSpec>,
    workgroupCount?: [ExprCode, ExprCode, ExprCode]
    shader: string,
}): KernelSpec {
    return {
        name: name,
        config: config,
        parameters: parameters,
        inputs: inputs,
        outputs: outputs,
        workgroupCount: workgroupCount,
        shader: shader,
    }
}

export const kernels: { [name: string]: KernelSpec } = {
    cumprod: _defaultKernel({
        name: "cumprod",
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        workgroupCount: ["parameters.outputSize / parameters.batchSize", "parameters.batchSize", 1],
        shader: `
            var batchStart: u32 = global_id.x * parameters.batchSize;
            var out: f32 = 1.0;
            for(var i: u32 = 0; i <= global_id.y; i++) {
                out = out * input[batchStart + i];
            }
            output[batchStart + global_id.y] = out;
        `
    }),
    pow: _defaultKernel({
        name: "pow",
        inputs: [
            { name: "a", shaderType: "array<f32>" },
            { name: "b", shaderType: "array<f32>" }
        ],
        shader: `output[global_id.x] = pow(a[global_id.x], b[global_id.y]);`
    }),
    cosh: _defaultKernel({
        name: "cosh",
        shader: `output[global_id.x] = cosh(input[global_id.x]);`
    }),
    cos: _defaultKernel({
        name: "sin", 
        shader: `output[global_id.x] = cos(input[global_id.x]);`
    }),
    sin: _defaultKernel({
        name: "sin", 
        shader: `output[global_id.x] = sin(input[global_id.x]);`
    }),
    sadd: _defaultKernel({
        name: "sadd",
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            },
            {
                name: "alpha",
                shaderType: "f32"
            }
        ],
        shader: `output[global_id.x] = input[global_id.x] + parameters.alpha;`
    }),
    ssub: _defaultKernel({
        name: "ssub",
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            },
            {
                name: "alpha",
                shaderType: "f32"
            }
        ],
        shader: `output[global_id.x] = input[global_id.x] - parameters.alpha;`
    }),
    smul: _defaultKernel({
        name: "smul",
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            },
            {
                name: "alpha",
                shaderType: "f32"
            }
        ],
        shader: `output[global_id.x] = input[global_id.x] * parameters.alpha;`
    }),
    sdiv: _defaultKernel({
        name: "sdiv",
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            },
            {
                name: "alpha",
                shaderType: "f32"
            }
        ],
        shader: `output[global_id.x] = input[global_id.x] / parameters.alpha;`
    }),
    min: _defaultKernel({
        name: "min",
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            },
            {
                name: "alpha",
                shaderType: "f32"
            }
        ],
        shader: `output[global_id.x] = min(input[global_id.x], parameters.alpha);`
    }),
    max: _defaultKernel({
        name: "max",
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            },
            {
                name: "alpha",
                shaderType: "f32"
            }
        ],
        shader: `output[global_id.x] = max(input[global_id.x], parameters.alpha);`
    }),
    gelu: _defaultKernel({
        name: "gelu",
        shader: `
        output[global_id.x] = 0.5 * input[global_id.x] * (1 + tanh(sqrt(2 / 3.14159265359) * (input[global_id.x] + 0.044715 * pow(input[global_id.x], 3))));
        `
    }),
    silu: _defaultKernel({
        name: "silu",
        shader: `output[global_id.x] = input[global_id.x] / (1.0 + exp(-input[global_id.x]));`
    }),
    sqrt: {
        name: "sqrt",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: 'array<f32>',
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", 1, 1],
        shader: `
            output[global_id.x] = sqrt(input[global_id.x]);
        `
    },
    add: {
        name: "add",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>"
            },
            {
                name: "b",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: 'array<f32>',
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", '1', '1'],
        shader: `
            output[global_id.x] = a[global_id.x] + b[global_id.x];
        `
    },
    sub: {
        name: "sub",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>"
            },
            {
                name: "b",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: 'array<f32>',
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", "1", "1"],
        shader: `
            output[global_id.x] = a[global_id.x] - b[global_id.x];
        `
    },
    mul: {
        name: "mul",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>"
            },
            {
                name: "b",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: 'array<f32>',
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", "1", '1'],
        shader: `
            output[global_id.x] = a[global_id.x] * b[global_id.x];
        `
    },
    div: {
        name: "div",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>"
            },
            {
                name: "b",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: 'array<f32>',
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", "1", "1"],
        shader: `
            output[global_id.x] = a[global_id.x] / b[global_id.x];
        `
    },
    find_index: {
        name: "find_index",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", "1", "1"],
        shader: `
            output[global_id.x] = f32(global_id.x);
        `
    },
    clamp: {
        name: "clamp",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "low",
                shaderType: "f32"
            },
            {
                name: "high",
                shaderType: "f32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        workgroupCount: ["parameters.outputSize", '1', '1'],
        shader: `
            output[global_id.x] = min(max(input[global_id.x], parameters.low), parameters.high);
        `
    },
    index: {
        name: "index",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "size",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            },
            {
                name: "index",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size"
            }
        ],
        workgroupCount: ["parameters.size", "1", "1"],
        shader: `
            output[global_id.x] = input[u32(index[global_id.x])];
        `
    },
    sum: {
        name: "sum",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "batches",
                shaderType: "u32"
            },
            {
                name: "batch_size",
                shaderType: "u32"
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size:"batches"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.batches", 1, 1],
        shader: `
            var batch_start: u32 = global_id.x * parameters.batch_size;

            var sum: f32 = 0;
            for(var i: u32 = 0; i < parameters.batch_size; i++) {
                sum += input[batch_start + i];
            }

            output[global_id.x] = sum;
        `
    },
    group_norm: {
        name: "group_norm",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "eps",
                shaderType: "f32"
            },
            {
                name: "groups",
                shaderType: "u32"
            },
            {
                name: "groupSize",
                shaderType: "u32"
            },
            {
                name: "channels",
                shaderType: "u32"
            },
            {
                name: "batches",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            },
            {
                name: "weight",
                shaderType: "array<f32>"
            },
            {
                name: "bias",
                shaderType: "array<f32>"
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.batches", "parameters.groups", "1"],
        shader: `
            var group_start = (global_id.x * parameters.groups + global_id.y) * parameters.groupSize;

            var mean: f32 = 0;
            var mean_sqrd: f32 = 0;

            for(var i: u32 = 0; i < parameters.groupSize; i++) {
                mean = mean + input[group_start + i];
                mean_sqrd = mean_sqrd + input[group_start + i] * input[group_start + i];
            }

            mean = mean / f32(parameters.groupSize);
            var variance: f32 = (mean_sqrd / f32(parameters.groupSize)) - (mean * mean) + parameters.eps;
            
            var channel: u32 = (global_id.x * parameters.groups + global_id.y / (parameters.groupSize / parameters.groups)) % parameters.channels;
            for(var i: u32 = 0; i < parameters.groupSize; i++) {
                output[group_start + i] = ((input[group_start + i] - mean) / sqrt(abs(variance))) * weight[channel] + bias[channel];
            }
            

        `
    },
    arange: {
        name: "arange",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "start",
                shaderType: "f32"
            },
            {
                name: "step",
                shaderType: "f32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", 1, 1],
        shader: `
            output[global_id.x] = parameters.step * f32(global_id.x) + parameters.start;
        `  
    },
    linspace: {
        name: "linspace",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "start",
                shaderType: "f32"
            },
            {
                name: "end",
                shaderType: "f32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", 1, 1],
        shader: `
            var diff = parameters.end - parameters.start;
            output[global_id.x] = diff * f32(global_id.x) / f32(parameters.outputSize + 1) + parameters.start;
        `  
    },
    transpose: {
        name: "transpose",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "dim0",
                shaderType: "u32"
            },
            {
                name: "dim1",
                shaderType: "u32"
            },
            {
                name: "batchSize",
                shaderType: "u32"
            },
            {
                name: "elSize",
                shaderType: "u32"
            },
            {
                name: "stride",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.batchSize / parameters.elSize", "parameters.outputSize / parameters.batchSize", "parameters.elSize"],
        shader: `
            var i: u32 = global_id.x / parameters.stride / parameters.dim1;
            var j: u32 = global_id.x % parameters.dim1;
            var k: u32 = global_id.x / parameters.dim1 % parameters.stride;

            var win_idx: u32 = j * parameters.dim0 * parameters.stride + k * parameters.dim0 + i;

            var output_idx: u32 = global_id.y * parameters.batchSize + win_idx * parameters.elSize + global_id.z;
            var input_idx: u32 = global_id.y * parameters.batchSize + global_id.x * parameters.elSize + global_id.z;
    
            output[output_idx] = input[input_idx];
        `
    },
    softmax: {
        name: "softmax",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "batches",
                shaderType: "u32"
            },
            {
                name: "batch_size",
                shaderType: "u32"
            },
            {
                name: "stride",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batches * batch_size * stride"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.batches", "parameters.batch_size", "parameters.stride"],
        shader: `
            var sum: f32 = 0;
            for(var i: u32 = 0; i < parameters.batch_size; i++) {
                sum += exp(input[global_id.x * parameters.batch_size + i * parameters.stride + global_id.z]);
            } 

            var idx: u32 = global_id.x * parameters.batch_size * parameters.stride + global_id.y * parameters.stride + global_id.z;
            output[idx] = exp(input[idx]) / sum;
        `
    },
    chunk: {
        name: "chunk",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "chunkSize",
                shaderType: "u32"
            },
            {
                name: "numChunks",
                shaderType: "u32"
            },
            {
                name: "offset",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.chunkSize", "parameters.outputSize / parameters.chunkSize", 1],
        shader: `
            var input_chunk_start: u32 = global_id.y * parameters.numChunks * parameters.chunkSize;
            var input_idx: u32 = input_chunk_start + parameters.offset * parameters.chunkSize + global_id.x;
            var output_idx: u32 = global_id.y * parameters.chunkSize + global_id.x;
            output[output_idx] = input[input_idx];
        `
    },
    box_muller: {
        name: "box_muller",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "mean",
                shaderType: "f32"
            },
            {
                name: "sdev",
                shaderType: "f32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 2, 1],
        workgroupCount: ["parameters.outputSize", "2", "1"],
        shader: `
            const pi = 3.1415;
            var u1: f32;
            var u2: f32;
            var idx: u32;
            if(global_id.y == 0) {
                u1 = input[global_id.x];
                u2 = input[global_id.x + (parameters.outputSize / 2)];
                idx = global_id.x;
            } else {
                u1 = input[global_id.x + (parameters.outputSize / 2)];
                u2 = input[global_id.x];
                idx = global_id.x + (parameters.outputSize / 2);
            }
            output[idx] = sqrt(abs(-2 * log(u1))) * cos(2 * pi * u2) * parameters.sdev + parameters.mean;
        `
    },
    upsample: {
        name: "upsample",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "mode",
                shaderType: "u32"
            },
            {
                name: "w_in",
                shaderType: "u32"
            },
            {
                name: "h_in",
                shaderType: "u32"
            },
            {
                name: "d_in",
                shaderType: "u32"
            },
            {
                name: "w_out",
                shaderType: "u32"
            },
            {
                name: "h_out",
                shaderType: "u32"
            },
            {
                name: "d_out",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize / parameters.h_out", "parameters.h_out", "parameters.d_out"],
        shader: `
            var output_idx = global_id.x * parameters.h_out * parameters.d_out + global_id.y * parameters.d_out + global_id.z;

            switch (parameters.mode) {
                case 0: { // nearest
                    var x_map = global_id.x * parameters.w_in / parameters.w_out;
                    var y_map = global_id.y * parameters.h_in / parameters.h_out;
                    var d_map = global_id.z * parameters.d_in / parameters.d_out;
                    var idx_map: u32 = x_map * parameters.h_in * parameters.d_in + y_map * parameters.d_in + d_map;
                    output[output_idx] = input[idx_map]; 
                    break;
                }
                case 1: { // linear
                    var scale_factor = parameters.w_out / parameters.w_in;
                    var offset = (scale_factor - 1) / 2;
                    var x_l = (global_id.x - offset) / scale_factor;
                    var x_r = x_l + 1;
                    if(x_r % parameters.w_in < abs(x_l) % parameters.w_in) {
                        if(global_id.x % parameters.w_out < scale_factor) {
                            output[output_idx] = input[x_r];
                        } else {
                            output[output_idx] = input[x_l];
                        }
                        return;
                    }
                    
                    var diff = (input[x_r] - input[x_l]) / f32(scale_factor);
                    output[output_idx] = input[x_l] + diff * f32((2*(global_id.x - offset)) % (2*scale_factor)) / 2.0;
                    break;
                }
                case 2: { // bilinear
                    var xscale_factor = parameters.w_out / parameters.w_in;
                    var xoffset = (xscale_factor - 1) / 2;
                    var yscale_factor = parameters.h_out / parameters.h_in;
                    var yoffset = (yscale_factor - 1) / 2;

                    var x1_map = (global_id.x - xoffset) / xscale_factor;
                    var x2_map = x1_map + 1;
                    var y1_map = (global_id.y - yoffset) / yscale_factor;
                    var y2_map = y1_map + 1;

                    /*
                    if(x_r % parameters.w_in < abs(x_l) % parameters.w_in) {
                        if(global_id.x % parameters.w_out < scale_factor) {
                            output[output_idx] = input[x_r];
                        } else {
                            output[output_idx] = input[x_l];
                        }
                        return;
                    }
                    */

                    var x_oob = x2_map % parameters.w_in < abs(x1_map) % parameters.w_in;
                    var y_oob = y2_map % parameters.w_in < abs(y1_map);
                    if(x_oob && y_oob) {
                        if(global_id.x % parameters.w_out < xscale_factor) {
                            if(global_id.y < yscale_factor) {
                                var idx: u32 = x2_map*parameters.h_in + y2_map;
                                output[output_idx] = input[idx];
                            } else {
                                var idx: u32 = x2_map*parameters.h_in + y1_map;
                                output[output_idx] = input[idx];
                            }
                        } else {
                            if(global_id.y < yscale_factor) {
                                var idx: u32 = x1_map*parameters.h_in + y2_map;
                                output[output_idx] = input[idx];
                            } else {
                                var idx: u32 = x1_map*parameters.h_in + y1_map;
                                output[output_idx] = input[idx];
                            }
                        }
                        //return;
                    } else if(x_oob && !y_oob) {
                        var p1Idx: u32;
                        var p2Idx: u32;
                        if(global_id.x % parameters.w_out < xscale_factor) {
                            p1Idx = x2_map*parameters.h_in + y1_map;
                            p2Idx = x2_map*parameters.h_in + y2_map;
                        } else {
                            p1Idx = x1_map*parameters.h_in + y1_map;
                            p2Idx = x1_map*parameters.h_in + y2_map;
                        }
                        var diff = (input[p2Idx] - input[p1Idx]) / f32(yscale_factor);
                        output[output_idx] = input[p1Idx] + diff * f32((2*(global_id.y - yoffset)) % (2*yscale_factor)) / 2.0;
                        //return;
                    } else if(!x_oob && y_oob) {
                        var p1Idx: u32;
                        var p2Idx: u32;
                        if(global_id.y < yscale_factor) {
                            p1Idx = x1_map*parameters.h_in + y2_map;
                            p2Idx = x2_map*parameters.h_in + y2_map;
                        } else {
                            p1Idx = x1_map*parameters.h_in + y1_map;
                            p2Idx = x2_map*parameters.h_in + y1_map;
                        }
                        var diff = (input[p2Idx] - input[p1Idx]) / f32(xscale_factor);
                        output[output_idx] = input[p1Idx] + diff * f32((2*(global_id.x - xoffset)) % (2*xscale_factor)) / 2.0;
                        //return;
                    }

                    // [ ... p1, x, ..., x, p2 ... ]
                    // [ ...        ...        ... ]
                    // [ ... p3, x, ..., x, p4 ... ]

                    var p1Idx = x1_map*parameters.h_in*parameters.d_in + y1_map*parameters.d_in;
                    var p2Idx = x1_map*parameters.h_in*parameters.d_in + y2_map*parameters.d_in;
                    var p3Idx = x2_map*parameters.h_in*parameters.d_in + y1_map*parameters.d_in;
                    var p4Idx = x2_map*parameters.h_in*parameters.d_in + y2_map*parameters.d_in;
                    var diff1 = (input[p2Idx] - input[p1Idx]) / f32(yscale_factor);
                    var diff2 = (input[p4Idx] - input[p3Idx]) / f32(yscale_factor);
                    var p5    = input[p1Idx] + diff1 * f32((2*(global_id.y - yoffset)) % (2*yscale_factor)) / 2.0;
                    var p6    = input[p3Idx] + diff2 * f32((2*(global_id.y - yoffset)) % (2*yscale_factor)) / 2.0;
                    var diff = (p6 - p5) / f32(xscale_factor);
                    
                    output[output_idx] = p5 + diff * f32((2*(global_id.x - xoffset)) % (2*xscale_factor)) / 2.0;
                    break;
                }
                case 3: { //bicubic
                    output[global_id.x] = 3;
                    break;
                }
                case 4: { //trilinear
                    output[global_id.x] = 4;
                    break;
                }
                default: {
                    output[global_id.x] = 5;
                    break;
                }
            }
        `
    },
    maxpool2d: {
        name:"maxpool2d",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "batches",
                shaderType: "u32"
            },
            {
                name: "kernel_size_x",
                shaderType: "u32"
            },
            {
                name: "kernel_size_y",
                shaderType: "u32"
            },
            {
                name: "stride_x",
                shaderType: "u32"
            },
            {
                name: "stride_y",
                shaderType: "u32"
            },
            {
                name: "input_height",
                shaderType: "u32"
            },
            {
                name: "input_width",
                shaderType: "u32"
            },
            {
                name: "output_height",
                shaderType: "u32"
            },
            {
                name: "output_width",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.batches", "parameters.output_height", "parameters.output_width"],
        shader: `
            var input_batch_idx = global_id.x * parameters.input_height * parameters.input_width;
            var output_batch_idx = global_id.x * parameters.output_height * parameters.output_width;

            var base_y = global_id.y * parameters.stride_y;
            var base_x = global_id.z * parameters.stride_x;

            var walker = input[input_batch_idx + base_y * parameters.input_width + base_x];
            for(var i: u32 = 0; i < parameters.kernel_size_y; i++) {
                for(var j: u32 = 0; j < parameters.kernel_size_x; j++) {
                    var input_idx: u32 = input_batch_idx + (base_y + i) * parameters.input_width + (base_x + j);
                    walker = max(walker, input[input_idx]);
                }
            }

            var output_idx: u32 = output_batch_idx + global_id.y * parameters.output_width + global_id.z;
            output[output_idx] = walker; 

            /*
            var pool_size = parameters.kernel_size_x * parameters.kernel_size_y;
            var row = global_id.x / parameters.row_len;
            var col = global_id.x % parameters.row_len;
            var row_start = parameters.stride_x * parameters.width_y * row;
            var pool_start = row_start + parameters.stride_y * col  + channel_offset;

            var walker = input[pool_start];
            for(var i: u32 = 0; i < parameters.kernel_size_x*parameters.dilation_x; i += parameters.dilation_x) {
                for(var j: u32 = 0; j < parameters.kernel_size_y*parameters.dilation_y; j += parameters.dilation_y) {
                    var idx: u32 = pool_start + j + parameters.width_y * i;
                    walker = max(walker, input[idx]);
                    //walker = pool_start;
                }
            }
            var output_channel_size = parameters.col_len * parameters.row_len;
            var output_idx = global_id.x + global_id.y * output_channel_size + global_id.z * output_channel_size * parameters.channel_width;
            output[output_idx] = walker;
            output[output_idx] = 1;
            */
        `
    },
    layernorm: {
        name: "layernorm",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "eps",
                shaderType: "f32"
            },
            {
                name: "norm_size",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>"
            },
            {
                name: "weight",
                shaderType: "array<f32>"
            },
            {
                name: "bias",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize / parameters.norm_size", "parameters.norm_size", 1],
        // y = (x - E[x]) / sqrt(Var[x] + eps) * y + B
        shader: `
            var norm_shape_idx = global_id.x * parameters.norm_size;
            var expectation = 0.0;
            var variance = 0.0;
            for(var i: u32 = 0; i < parameters.norm_size; i++) {
                expectation = expectation + input[norm_shape_idx + i];
                variance = variance + input[norm_shape_idx + i] * input[norm_shape_idx + i];
            }
            expectation = expectation / f32(parameters.norm_size);
            variance = variance / f32(parameters.norm_size);
            variance = abs(variance - pow(expectation, 2));

            output[norm_shape_idx + global_id.y] = (input[norm_shape_idx + global_id.y] - expectation) / sqrt(abs(variance + parameters.eps)) * weight[global_id.y] + bias[global_id.y];
        `
    },
    cat: {
        name: "cat",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "dim",
                shaderType: "u32"
            },
            {
                name: "part",
                shaderType: "u32"
            },
            {
                name: "stride",
                shaderType: "u32"
            },
            {
                name: "outputSize",
                shaderType: "u32"
            }
            
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>"
            },
            {
                name: "b",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        //workgroupSize: [1, 1, 1],
        workgroupCount: ["parameters.outputSize", 1, 1],
        shader: `
            if(global_id.x % parameters.stride < parameters.part) { 
                var idx = ((global_id.x / parameters.stride) * parameters.part) + (global_id.x % parameters.stride);
                output[global_id.x] = a[idx];
            }
            else { 
                var idx = ((global_id.x / parameters.stride) * (parameters.stride - parameters.part)) + (global_id.x % parameters.stride) - parameters.part;
                output[global_id.x] = b[idx];
            }
        `
    },
    conv2d: {
        name: "conv2d",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32",
            },
            {
                name: "inputChannels",
                shaderType: "u32",
            },
            {
                name: "outputChannels",
                shaderType: "u32",
            },
            {
                name: "inputHeight",
                shaderType: "u32",
            },
            {
                name: "inputWidth",
                shaderType: "u32",
            },
            {
                name: "kernelHeight",
                shaderType: "u32",
            },
            {
                name: "kernelWidth",
                shaderType: "u32",
            },
            {
                name: "outputHeight",
                shaderType: "u32",
            },
            {
                name: "outputWidth",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "weight",
                shaderType: "array<f32>",
            },
            {
                name: "bias",
                shaderType: "array<f32>"
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * outputChannels * outputHeight * outputWidth",
            },
        ],
        //workgroupCount: ["parameters.outputWidth", "parameters.outputHeight", '1'],
        workgroupCount: ["parameters.outputChannels", "parameters.outputHeight", "parameters.outputWidth"],
        shader: `

            var max_input_idx: u32 = parameters.batchSize * parameters.inputChannels * parameters.inputHeight * parameters.inputWidth;
            var max_weight_idx: u32 = parameters.outputChannels * parameters.inputChannels * parameters.kernelHeight * parameters.kernelWidth;

            for(var batch: u32 = 0; batch < parameters.batchSize; batch++) {
                var input_batch_idx: u32 = batch * parameters.inputChannels * parameters.inputHeight * parameters.inputWidth;
                var weight_batch_idx: u32 = batch * parameters.outputChannels * parameters.inputChannels * parameters.kernelHeight * parameters.kernelWidth;
                var output_batch_idx: u32 = batch * parameters.outputChannels * parameters.outputHeight * parameters.outputWidth;
                
                
                var weight_out_channel_idx: u32 = weight_batch_idx + global_id.x * parameters.inputChannels * parameters.kernelHeight * parameters.kernelWidth;

                var output_channel_idx: u32 = output_batch_idx + global_id.x * parameters.outputHeight * parameters.outputWidth;
                var output_idx: u32 = output_channel_idx + global_id.y * parameters.outputWidth + global_id.z;

                var result: f32 = 0;
                for(var i: u32 = 0; i < parameters.inputChannels; i++) {
                    var input_channel_idx: u32 = input_batch_idx + i * parameters.inputHeight * parameters.inputWidth;
                    var weight_channel_idx: u32 = weight_out_channel_idx + i * parameters.kernelHeight * parameters.kernelWidth;

                    for(var j: u32 = 0; j < parameters.kernelHeight; j++) {
                        for(var k: u32 = 0; k < parameters.kernelWidth; k++) {
                            var input_idx: u32 = input_channel_idx + (global_id.y + j) * parameters.inputWidth + (global_id.z + k);
                            var weight_idx: u32 = weight_channel_idx + j * parameters.kernelWidth + k;
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                var bias_idx: u32 = batch * parameters.outputChannels + global_id.x;
                output[output_idx] = result + bias[bias_idx];
            }
        `
    },
    mm: {
        name: "mm",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "resultRows",
                shaderType: "u32",
            },
            {
                name: "resultCols",
                shaderType: "u32",
            },
            {
                name: "innerDim",
                shaderType: "u32",
            },
            {
                name: "alpha",
                shaderType: "f32",
            },
            {
                name: "batches",
                shaderType: "u32"
            }
        ],
        inputs: [
            {
                name: "firstMatrix",
                shaderType: "array<f32>",
            },
            {
                name: "secondMatrix",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "resultMatrix",
                shaderType: "array<f32>",
                size: "resultRows * resultCols * batches",
            },
        ],
        workgroupCount: ["parameters.resultRows", "parameters.resultCols", "parameters.batches"],
        shader: `
            var result = 0.0;
            for (var i = 0u; i < parameters.innerDim; i = i + 1u) {
                let a = global_id.x * parameters.innerDim + i               + parameters.resultRows * parameters.innerDim * global_id.z;
                let b = i * parameters.resultCols + global_id.y             + parameters.resultCols * parameters.innerDim * global_id.z;
                result = result + firstMatrix[a] * secondMatrix[b];
            }
            let index = global_id.y + global_id.x * parameters.resultCols  + parameters.resultRows * parameters.resultCols * global_id.z;
            resultMatrix[index] = result;
`
    },
    sumDim: {
        name: "sumDim",
        config: [
            {
                name: "dtype",
            },
            {
                name: "workgroupSize",
            },
        ],
        parameters: [
            {
                name: "size",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "dimToSum",
                shaderType: "u32",
            },
            {
                name: "inputShape",
                shaderType: "vec3<u32>",
            },
            {
                name: "inputStrides",
                shaderType: "vec3<u32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        //workgroupSize: ["workgroupSize", "workgroupSize", "workgroupSize"],
        workgroupCount: ['1', "1", '1'],
        shader: `
        // Global index flattening for the reformatted 3D tensor
        var flatGlobalId: u32 = global_id.x * parameters.inputStrides.x + global_id.y * parameters.inputStrides.y + global_id.z * parameters.inputStrides.z;
    
        // Initialize sum
        var sum: f32 = 0.0;
    
        let numReductions: u32 = parameters.inputShape.y;
    
        // Sum reduction
        for (var i: u32 = 0; i < numReductions; i = i + 1) {
            // Compute the input index by adding the reduction offset to the current flat global index
            var dataIndex: u32 = flatGlobalId + i * parameters.inputStrides.y;
    
            if (dataIndex < input.length()) {
                // Accumulate the input value into sum
                sum = sum + input[dataIndex];
            }
        }
    
        // Write the reduced sum value to output tensor
        if (flatGlobalId < output.length()) {
            output[flatGlobalId] = sum;
        }
    `
    },
};
