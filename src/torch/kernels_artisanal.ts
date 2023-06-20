import { KernelSpec } from "./kernel";

export const kernels: { [name: string]: KernelSpec } = {
    group_norm: {
        name: "group_norm",
        config: [
            {
                name: "dtype"
            }
        ],
        parameters: [
            {
                name: "groupSize",
                shaderType: "u32"
            },
            {
                name: "eps",
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
        workgroupSize: [1, 1, 1],
        workgroupCount: ["outputSize", 1, 1],
        shader: `
            if(global_id.x > parameters.outputSize) return;

            const group = floor(global_id.x / parameters.groupSize);
            const group_start = group * parameters.groupSize;

            var mean = 0;
            var mean_sqrd = 0;

            for(let i = group * parameters.groupSize; i < (group + 1) * parameters.groupSize; i++) {
                mean = mean + input[i];
                mean_sqrd = mean_sqrd + input[i] * input[i];
            }

            mean = mean / parameters.groupSize;
            const std = sqrt( (mean_sqrd / parameters.groupSize) - (mean * mean) + [parameters.eps]);

            output[global_id.x] = (input[global_id.x] - mean) / std * weight[group] + bias[group];

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
        workgroupSize: [8, 1, 1],
        workgroupCount: ["outputSize / 8", 1, 1],
        shader: `
            if(global_id.x >= parameters.outputSize) return;

            output[global_id.x] = parameters.step * global_id.x + parameters.start;
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
        workgroupSize: [8, 1, 1],
        workgroupCount: ["outputSize / 8", 1, 1],
        shader: `
            if(global_id.x >= parameters.outputSize) return;

            const diff = parameters.end - parameters.start;
            output[global_id.x] = diff * global_id.x / (parameters.outputSize - 1) + parameters.start;
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
                size: "outputSize * batchSize * elSize"
            }
        ],
        workgroupSize: [1, 1, 1],
        workgroupCount: ["batchSize / elSize", "outputSize / batchSize", "elSize"],
        shader: `
            if(global_id.x > parameters.batchSize / parameters.elSize) {
                return;
            }

            const i = floor(global_id.x / parameters.stride / parameters.dim1);
            const j = global_id.x % parameters.dim1;
            const k = floor(global_id.x / parameters.dim1) % parameters.stride;

            const win_idx = j * parameters.dim0 * parameters.stride + k * parameters.dim0 + i;

            const output_idx = global_id.y * parameters.batchSize + win_idx * parameters.elSize + global_id.z;
            const input_idx = global_id.y * parameters.batchSize + global_id.x * parameters.elSize + global_id.z
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
            },
            {
                name: "sums",
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
        workgroupSize: [4, 4, 1],
        workgroupCount: ["stride / 4", "outputSize / stride / 4", 1],
        shader: `
            if(global_id.x > stride || global_id.y > floor(outputSize / stride)) {
                return;
            }

            const idx = global_id.x + global_id.y*stride;
            output[idx] = input[idx] / sum[global_id.y];
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
                name: "stride",
                shaderType: "u32"
            },
            {
                name: "strides",
                shaderType: "u32"
            },
            {
                name: "offset",
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
                size: "chunkSize * strides"
            }
        ],
        workgroupSize: [4, 4, 1],
        workgroupCount: ["chunkSize / 4", "strides / 4", 1],
        shader: `
            if (global_id.x >= parameters.chunkSize || global_id.y >= parameters.strides) {
                return;
            }

            const chunk_idx = global_id.y*parameters.chunkSize;
            output[global_id.x + chunk_idx] = input[global_id.x + chunk_idx*parameters.stride];
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
                name: "std",
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
        workgroupSize: [4, 2, 1],
        workgroupCount: ["outputSize / 8", 2, 1],
        shader: `
            if (global_id.x + global_id.y*(parameters.outputSize / 2) >= parameters.outputSize) {
                return;
            }

            const pi = 3.1415
            let u1;
            let u2;
            let idx;
            if(global_id.y == 0) {
                u1 = input[global_id.x];
                u2 = input[global_id.x + (parameters.outputSize / 2)];
                idx = global_id.x;
            } else {
                u1 = input[global_id.x + (parameters.outputSize / 2)];
                u2 = input[global_id.x];
                idx = global_id.x + (parameters.outputSize / 2);
            }
            output[idx] = sqrt(-2 * log(u1)) * cos(2 * pi * u2) * parameters.std + parameters.mean;
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
        workgroupSize: [8, 1, 1],
        workgroupCount: ["outputSize / h_out / 8", "h_out", "d_out"],
        shader: `
            //inputIdx = x*parameters.h_in*parameters.d_in + y*parameters.d_in + z;

            const output_idx = global_id.x * parameters.h_out * parameters.d_out + global_id.y * parameters.d_out + global_id.z;
            if (output_idx >= parameters.outputSize) {
                return;
            }

            switch (parameters.mode) {
                case 0: { // nearest
                    const x_map = floor(global_id.x * parameters.w_in / parameters.w_out);
                    const y_map = floor(global_id.y * parameters.h_in / parameters.h_out);
                    const d_map = floor(global_id.z * parameters.d_in / parameters.d_out);
                    const idx_map = x_map * parameters.h_in * parameters.d_in + y_map * parameters.d_in + d_map;
                    output[output_idx] = input[idx_map]; 
                    break;
                }
                case 1: { // linear
                    const scale_factor = parameters.w_out / parameters.w_in;
                    const offset = (scale_factor - 1) / 2;
                    const x_l = floor((global_id.x - offset) / scale_factor);
                    const x_r = x_l + 1
                    if(x_r % parameters.w_in < abs(x_l) % parameters.w_in) {
                        if(global_id.x % parameters.w_out < scale_factor) {
                            output[output_idx] = input[x_r];
                        } else {
                            output[output_idx] = input[x_l];
                        }
                        return;
                    }
                    
                    const diff = (input[x_r] - input[x_l]) / scale_factor;
                    output[output_idx] = input[x_l] + diff * ((2*(global_id.x - offset)) % (2*scale_factor))/2;
                    break;
                }
                case 2: { // bilinear
                    const xscale_factor = parameters.w_out / parameters.w_in;
                    const xoffset = (xscale_factor - 1) / 2;
                    const yscale_factor = parameters.h_out / parameters.h_in;
                    const yoffset = (yscale_factor - 1) / 2;

                    const x1_map = floor((global_id.x - xoffset) / xscale_factor);
                    const x2_map = floor(x1_map + 1);
                    const y1_map = floor((global_id.y - yoffset) / yscale_factor);
                    const y2_map = floor(y1_map + 1);

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

                    const x_oob = x2_map % parameters.w_in < abs(x1_map) % parameters.w_in;
                    const y_oob = y2_map % parameters.w_in < abs(y1_map);
                    if(x_oob && y_oob) {
                        if(global_id.x % parameters.w_out < xscale_factor) {
                            if(global_id.y < yscale_factor) {
                                const idx = x2_map*parameters.h_in + y2_map;
                                output[output_idx] = input[idx];
                            } else {
                                const idx = x2_map*parameters.h_in + y1_map;
                                output[output_idx] = input[idx];
                            }
                        } else {
                            if(global_id.y < yscale_factor) {
                                const idx = x1_map*parameters.h_in + y2_map;
                                output[output_idx] = input[idx];
                            } else {
                                const idx = x1_map*parameters.h_in + y1_map;
                                output[output_idx] = input[idx];
                            }
                        }
                        return;
                    } else if(x_oob && !y_oob) {
                        let p1Idx;
                        let p2Idx;
                        if(global_id.x % parameters.w_out < xscale_factor) {
                            p1Idx = x2_map*parameters.h_in + y1_map;
                            p2Idx = x2_map*parameters.h_in + y2_map;
                        } else {
                            p1Idx = x1_map*parameters.h_in + y1_map;
                            p2Idx = x1_map*parameters.h_in + y2_map;
                        }
                        const diff = (input[p2Idx] - input[p1Idx]) / yscale_factor;
                        output[output_idx] = input[p1Idx] + diff * ((2*(global_id.y - yoffset)) % (2*yscale_factor))/2;
                        return;
                    } else if(!x_oob && y_oob) {
                        let p1Idx;
                        let p2Idx;
                        if(global_id.y < yscale_factor) {
                            p1Idx = x1_map*parameters.h_in + y2_map;
                            p2Idx = x2_map*parameters.h_in + y2_map;
                        } else {
                            p1Idx = x1_map*parameters.h_in + y1_map;
                            p2Idx = x2_map*parameters.h_in + y1_map;
                        }
                        const diff = (input[p2Idx] - input[p1Idx]) / xscale_factor;
                        output[output_idx] = input[p1Idx] + diff * ((2*(global_id.x - xoffset)) % (2*xscale_factor))/2;
                        return;
                    }

                    // [ ... p1, x, ..., x, p2 ... ]
                    // [ ...        ...        ... ]
                    // [ ... p3, x, ..., x, p4 ... ]

                    const p1Idx = x1_map*parameters.h_in*parameters.d_in + y1_map*parameters.d_in;
                    const p2Idx = x1_map*parameters.h_in*parameters.d_in + y2_map*parameters.d_in;
                    const p3Idx = x2_map*parameters.h_in*parameters.d_in + y1_map*parameters.d_in;
                    const p4Idx = x2_map*parameters.h_in*parameters.d_in + y2_map*parameters.d_in;
                    const diff1 = (input[p2Idx] - input[p1Idx]) / yscale_factor;
                    const diff2 = (input[p4Idx] - input[p3Idx]) / yscale_factor;
                    const p5    = input[p1Idx] + diff1 * ((2*(global_id.y - yoffset)) % (2*yscale_factor))/2;
                    const p6    = input[p3Idx] + diff2 * ((2*(global_id.y - yoffset)) % (2*yscale_factor))/2;
                    const diff = (p6 - p5) / xscale_factor;
                    
                    output[output_idx] = p5 + diff * ((2*(global_id.x - xoffset)) % (2*xscale_factor))/2;
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
                name: "padding_x",
                shaderType: "u32"
            },
            {
                name: "padding_y",
                shaderType: "u32"
            },
            {
                name: "dilation_x",
                shaderType: "u32"
            },
            {
                name: "dilation_y",
                shaderType: "u32"
            },
            {
                name: "width_y",
                shaderType: "u32"
            },
            {
                name: "row_len",
                shaderType: "u32"
            },
            {
                name: "col_len",
                shaderType: "u32"
            },
            {
                name: "channel_width",
                shaderType: "u32"
            },
            {
                name: "channel_height",
                shaderType: "u32"
            },
            {
                name: "channel_size",
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
        workgroupSize: [8, 1, 1],
        workgroupCount: ["col_len * row_len / 8", "channel_width", "channel_height"],
        shader: `
            if (global_id.x >= parameters.outputSize) {
                return;
            }

            var channel_offset = parameters.channel_size * (global_id.y + global_id.z * parameters.channel_width);

            var pool_size = parameters.kernel_size_x * parameters.kernel_size_y;
            var row = floor(global_id.x / parameters.row_len);
            var col = global_id.x % parameters.row_len;
            var row_start = parameters.stride_x * parameters.width_y * row;
            var pool_start = row_start + parameters.stride_y * col  + channel_offset;

            var walker = input[pool_start];
            for(var i = 0; i < parameters.kernel_size_x*parameters.dilation_x; i += parameters.dilation_x) {
                for(var j = 0; j < parameters.kernel_size_y*parameters.dilation_y; j += parameters.dilation_y) {
                    const idx = pool_start + j + parameters.width_y * i;
                    walker = max(walker, input[idx]);
                    //walker = pool_start;
                }
            }
            var output_channel_size = parameters.col_len * parameters.row_len;
            var output_idx = global_id.x + global_id.y * output_channel_size + global_id.z * output_channel_size * parameters.channel_width;
            output[output_idx] = walker;
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
                name: "gamma",
                shaderType: "f32"
            },
            {
                name: "beta",
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
            }
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize"
            }
        ],
        workgroupSize: [8, 1, 1],
        workgroupCount: ["outputSize / 8", 1, 1],
        // y = (x - E[x]) / sqrt(Var[x] + eps) * y + B
        shader: `
            if (global_id.x >= parameters.outputSize) {
                return;
            }
            const norm_shape_idx = floor(global_id.x / parameters.norm_size) * parameters.norm_size
            var expectation = 0.0;
            var variance = 0.0;
            for(var i = 0; i < parameters.norm_size; i++) {
                expectation += input[norm_shape_idx + i];
                variance += pow(input[norm_shape_idx + i], 2)
            }
            expectation = expectation / parameters.norm_size;
            variance = variance / parameters.norm_size;
            variance = abs(variance - pow(expectation, 2));

            output[global_id.x] = (input[global_id.x] - expectation)
                / sqrt(variance + parameters.eps)
                * parameters.gamma + parameters.beta;
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
        workgroupSize: [8, 1, 1],
        workgroupCount: ["outputSize / 8", 1, 1],
        shader: `
            if (global_id.x >= parameters.outputSize) {
                return;
            }
            if(global_id.x % parameters.stride < parameters.part) { 
                const idx = (floor(global_id.x / parameters.stride) * parameters.part) + (global_id.x % parameters.stride)
                output[global_id.x] = a[idx] 
            }
            else { 
                const idx = (floor(global_id.x / parameters.stride) * (parameters.stride - parameters.part)) + (global_id.x % parameters.stride) - parameters.part
                output[global_id.x] = b[idx] 
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
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * outputChannels * outputHeight * outputWidth",
            },
        ],
        workgroupSize: [4, 4, 1],
        workgroupCount: ["outputWidth/4", "outputHeight/4", 1],
        shader: `
    if (global_id.x >= parameters.outputWidth || global_id.y >= parameters.outputHeight) {
        return;
    }
    // input shape = [B, C, H, W]
    for (var batch = 0u; batch < parameters.batchSize; batch++) {
        for (var outputChannel = 0u; outputChannel < parameters.outputChannels; outputChannel++) {
            var result = 0.0;
            // Do the convolution
            for (var inputChannel = 0u; inputChannel < parameters.inputChannels; inputChannel++) {
                for (var kernelY = 0u; kernelY < parameters.kernelHeight; kernelY++) {
                    for (var kernelX = 0u; kernelX < parameters.kernelWidth; kernelX++) {
                        var inputY = global_id.y + kernelY;
                        var inputX = global_id.x + kernelX;
                        var inputIndex =
                            batch * parameters.inputChannels * parameters.inputHeight * parameters.inputWidth +
                            inputChannel * parameters.inputHeight * parameters.inputWidth +
                            inputY * parameters.inputWidth +
                            inputX;
                        var kernelIndex =
                            outputChannel * parameters.inputChannels * parameters.kernelHeight * parameters.kernelWidth +
                            inputChannel * parameters.kernelHeight * parameters.kernelWidth +
                            kernelY * parameters.kernelWidth +
                            kernelX;
                        result = result + input[inputIndex] * weight[kernelIndex];
                    }
                }
            }
            // Output
            let outputIndex = 
                batch * parameters.outputChannels * parameters.outputHeight * parameters.outputWidth +
                outputChannel * parameters.outputHeight * parameters.outputWidth +
                global_id.y * parameters.outputWidth +
                global_id.x;
            output[outputIndex] = result;
        }
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
                size: "resultRows * resultCols",
            },
        ],
        workgroupSize: [8, 8, 1],
        workgroupCount: ["resultRows/8", "resultCols/8", 1],
        shader: `
    if (global_id.x >= parameters.resultRows || global_id.y >= parameters.resultCols) {
        return;
    }
    var result = 0.0;
    for (var i = 0u; i < parameters.innerDim; i = i + 1u) {
        let a = global_id.x * parameters.innerDim + i;
        let b = i * parameters.resultCols + global_id.y;
        result = result + firstMatrix[a] * secondMatrix[b];
    }
    let index = global_id.y + global_id.x * parameters.resultCols;
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
        workgroupSize: ["workgroupSize", "workgroupSize", "workgroupSize"],
        workgroupCount: [1, 1, 1],
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
