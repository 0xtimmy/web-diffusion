import { KernelSpec } from "./kernel";

export const kernels: { [name: string]: KernelSpec } = {
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
