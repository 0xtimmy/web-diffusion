import { Device } from "./device";
import { EvalEnv } from "./expr";
import {
    Kernel,
    KernelConfig,
    KernelInputSpec,
    KernelOutputSpec,
    KernelParamsInput,
    KernelSpec,
    getKernelShaderCode,
    getShaderTypeElementByteSize,
} from "./kernel";

export class KernelWebGPU extends Kernel {
    private _gpuDevice: GPUDevice;
    private _bindGroupLayout: GPUBindGroupLayout;
    private _computePipeline: GPUComputePipeline;
    private _shaderCode: string;
    private _workgroupSizes: { x: number, y: number, z: number };

    constructor(spec: KernelSpec, config: KernelConfig, device: Device, params: null | any =null) {
        super(spec, config, device);
        const gpuDevice = (device as any).gpuDevice;
        if (!gpuDevice) {
            throw new Error("Cannot create a GPU kernel without a GPU device");
        }
        this._gpuDevice = gpuDevice;
        let bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = [];
        let bindingIndex = 0;
        for (let i = 0; i < spec.inputs.length; i++, bindingIndex++) {
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "read-only-storage" as GPUBufferBindingType,
                },
            });
        }
        for (let i = 0; i < spec.outputs.length; i++, bindingIndex++) {
            bindGroupLayoutEntries.push({
                binding: bindingIndex,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: "storage" as GPUBufferBindingType,
                },
            });
        }
        bindGroupLayoutEntries.push({
            binding: bindingIndex,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
                type: "read-only-storage" as GPUBufferBindingType,
            },
        });
        this._bindGroupLayout = gpuDevice.createBindGroupLayout({
            entries: bindGroupLayoutEntries,
        });

        //console.log("constructing kernel with params: ", params);
        const workgroupCounts: [number, number, number] = params == null ? [1, 1, 1] : this.getWorkgroupCounts(this.getRunEnv(params));
        const workload = getKernelShaderCode(spec, config, workgroupCounts, this._gpuDevice);

        this._shaderCode = workload.code;
        this._workgroupSizes = workload.workgroupSizes;

        const shaderModule = gpuDevice.createShaderModule({
            code: this._shaderCode,
        });
        this._computePipeline = gpuDevice.createComputePipeline({
            layout: gpuDevice.createPipelineLayout({
                bindGroupLayouts: [this._bindGroupLayout],
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });
    }
    run(
        inputs: GPUBuffer[],
        parameters: KernelParamsInput,
        outputs?: GPUBuffer[]
    ): GPUBuffer[] {
        //console.log("run gpu kernel", this.key);

        // Build the parameter environment
        const env: EvalEnv = this.getRunEnv(parameters);

        // Get the workgroup counts
        const [workgroupCountX, workgroupCountY, workgroupCountZ] =
            this.getWorkgroupCounts(env);

        // Get input buffers with storage usage
        const storageInputs = this.spec.inputs.map((input, i) =>
            this.getStorageInputBuffer(
                input,
                inputs[i] ? inputs[i] : null,
                i,
                env
            )
        );

        // Get output buffers with storage usage
        const storageOutputs = this.spec.outputs.map((output, i) =>
            this.getStorageOutputBuffer(
                output,
                outputs ? outputs[i] : null,
                i,
                env
            )
        );

        const maxCountPerDim = this._gpuDevice.limits.maxComputeWorkgroupsPerDimension;
        
        for(let i = 0; i < workgroupCountX;) {
            for(let j = 0; j < workgroupCountY;) {
                for(let k = 0; k <workgroupCountZ;) {
                    //console.log(`running kernel: ${this.key} with offset <${i}, ${j}, ${k}>`);
                    let paramsBufferSize = 3*getShaderTypeElementByteSize("u32");
                    for (let i = 0; i < this.spec.parameters.length; i++) {
                        const param = this.spec.parameters[i];
                        paramsBufferSize += getShaderTypeElementByteSize(param.shaderType);
                    }

                    // Build the params buffer
                    const paramsBuffer = this._gpuDevice.createBuffer({
                        mappedAtCreation: true,
                        size: paramsBufferSize,
                        usage: GPUBufferUsage.STORAGE,
                    });
                    const paramsArrayBuffer = paramsBuffer.getMappedRange();
                    let offsetsArray = new Uint32Array(paramsArrayBuffer);
                    offsetsArray[0] = i;
                    offsetsArray[1] = j;
                    offsetsArray[2] = k;

                    for (let paramDtype of ["u32", "f32"]) {
                        let paramsArray = new (
                            paramDtype === "u32" ? Uint32Array : Float32Array
                        )(paramsArrayBuffer);
                        for (let i = 0; i < this.spec.parameters.length; i++) {
                            const param = this.spec.parameters[i];
                            if (param.shaderType === paramDtype) {
                                paramsArray[i+3] = +env[param.name];
                            }
                        }
                    }
                    paramsBuffer.unmap();

                    // Bind the buffers
                    const bindGroup = this.createBindGroup(
                        storageInputs,
                        paramsBuffer,
                        storageOutputs
                    );

                    // Start a new command encoder
                    const commandEncoder = this._gpuDevice.createCommandEncoder();

                    const countX = Math.min(workgroupCountX - i, maxCountPerDim);
                    const countY = Math.min(workgroupCountY - j, maxCountPerDim);
                    const countZ = Math.min(workgroupCountZ - k, maxCountPerDim);
                    /*                 
                    console.log(`
    running: {
        x: ${countX} workgroups @ size ${this._workgroupSizes.x},
        y: ${countY} workgroups @ size ${this._workgroupSizes.y},
        z: ${countZ} workgroups @ size ${this._workgroupSizes.z},
    }
    with code:
    ${this._shaderCode}
                    `);
                    */

                    // Encode the kernel using pass encoder
                    const passEncoder = commandEncoder.beginComputePass();
                    passEncoder.setPipeline(this._computePipeline);
                    passEncoder.setBindGroup(0, bindGroup);
                    passEncoder.dispatchWorkgroups(
                        Math.ceil(countX / this._workgroupSizes.x),
                        Math.ceil(countY / this._workgroupSizes.y),
                        Math.ceil(countZ / this._workgroupSizes.z),
                    );
                    passEncoder.end();

                    // Submit GPU commands
                    const gpuCommands = commandEncoder.finish();
                    this._gpuDevice.queue.submit([gpuCommands]);

                    i += countX;
                    j += countY;
                    k += countZ;
                    
                }
            }
        }

        // Return the storage output buffers
        return storageOutputs;
    }
    private getStorageInputBuffer(
        inputSpec: KernelInputSpec,
        providedInput: GPUBuffer | null,
        inputIndex: number,
        env: EvalEnv
    ): GPUBuffer {
        if (providedInput === null) {
            throw new Error(
                `Missing input buffer #${inputIndex} (out of ${this.spec.inputs.length}) named "${inputSpec.name}" in kernel "${this.key}"`
            );
        }
        if (providedInput.usage & GPUBufferUsage.STORAGE) {
            providedInput.unmap();
            return providedInput;
        } else {
            throw new Error("Provided input buffer is not a storage buffer");
        }
    }
    private getStorageOutputBuffer(
        outputSpec: KernelOutputSpec,
        providedOutput: GPUBuffer | null,
        outputIndex: number,
        env: EvalEnv
    ): GPUBuffer {
        if (providedOutput !== null) {
            if (providedOutput.usage & GPUBufferUsage.STORAGE) {
                providedOutput.unmap();
                return providedOutput;
            } else {
                throw new Error(
                    "Provided output buffer is not a storage buffer"
                );
            }
        } else {
            const outputElementByteSize = getShaderTypeElementByteSize(
                outputSpec.shaderType
            );
            const outputElementCount = Math.ceil(
                this._outputSizeFuncs[outputIndex](env)
            );
            // console.log("output size", outputElementCount, outputElementByteSize);
            const outputBufferSize = outputElementByteSize * outputElementCount;
            const outputBuffer = this._gpuDevice.createBuffer({
                mappedAtCreation: false,
                size: outputBufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            return outputBuffer;
        }
    }
    private createBindGroup(
        inputBuffers: GPUBuffer[],
        paramsBuffer: GPUBuffer,
        outputBuffers: GPUBuffer[]
    ): GPUBindGroup {
        const entries: GPUBindGroupEntry[] = [];

        let bindingIndex = 0;
        for (let i = 0; i < inputBuffers.length; i++, bindingIndex++) {
            if(inputBuffers[i].size > 134217728) throw new Error("Buffer size too big!");
            //console.log("Buffer size:", inputBuffers[i].size);
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: inputBuffers[i],
                },
            });
        }
        for (let i = 0; i < this.spec.outputs.length; i++, bindingIndex++) {
            if(outputBuffers[i].size > 134217728) throw new Error("Buffer size too big!");
            entries.push({
                binding: bindingIndex,
                resource: {
                    buffer: outputBuffers[i],
                },
            });
        }
        entries.push({
            binding: bindingIndex,
            resource: {
                buffer: paramsBuffer,
            },
        });
        const bindGroup = this._gpuDevice.createBindGroup({
            layout: this._bindGroupLayout,
            entries: entries,
        });
        return bindGroup;
    }
}
