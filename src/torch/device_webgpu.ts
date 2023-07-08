import { Device } from "./device";
import type { ATypedArray, Dtype } from "./dtype";
import { GPUBufferStorage, UntypedStorage } from "./storage";
import type { Kernel, KernelConfig, KernelSpec } from "./kernel";
import { KernelWebGPU } from "./kernel_webgpu";

export class DeviceWebGPU extends Device {
    private _device: GPUDevice;

    _alloced_buffers: { [key: string]: GPUBuffer };
    private _buffer_counter: number;

    get gpuDevice(): GPUDevice {
        return this._device;
    }
    constructor(id: string, adapter: GPUAdapter, device: GPUDevice) {
        super(id, "webgpu");

        this._alloced_buffers = {};
        this._buffer_counter = 0;

        this._device = device;
        this._device.pushErrorScope("out-of-memory");
        this._device.addEventListener("uncapturederror", (error: any) => {
            console.error("Uncaptured error emited from GPU: ", error);
        })
        this._device.onuncapturederror = (error: any) => {
            console.error("Uncaptured error emited from GPU: ", error);
        };
        this._device.lost.then((lostInfo: GPUDeviceLostInfo) => {
            console.warn("lost gpu info:", lostInfo);
        })

    }
    alloc(byteSize: number): GPUBufferStorage {
        return new GPUBufferStorage(
            byteSize,
            this._device,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        );
    }
    createKernel(spec: KernelSpec, config: KernelConfig, params=null): Kernel {
        return new KernelWebGPU(spec, config, this, params);
    }
    getStorageFromKernel(storage: ATypedArray | GPUBuffer): UntypedStorage {
        if (storage instanceof GPUBuffer) {
            return new GPUBufferStorage(storage, this.gpuDevice);
        }
        throw new Error(
            `Cannot wrap buffer of type ${storage.constructor.name} to get GPU storage`
        );
    }
    getBufferForKernel(
        storage: UntypedStorage,
        dtype: Dtype
    ): ATypedArray | GPUBuffer {
        if (storage instanceof GPUBufferStorage) {
            return storage.gpuBuffer;
        }
        throw new Error(
            `Cannot unwrap storage of type ${storage.constructor.name} to get GPU buffer`
        );
    }
    createBuffer({
        label,
        mappedAtCreation=false,
        size,
        usage
    }: {
        label?: string,
        mappedAtCreation: boolean,
        size: number,
        usage: any
    }): GPUBuffer {
        const buf =  this._device.createBuffer({
            label: `${this._buffer_counter} - ${label}`,
            mappedAtCreation: mappedAtCreation,
            size: size,
            usage: usage
        });
        this._alloced_buffers[buf.label] = buf;
        this._buffer_counter++;
        return buf;
    }
    destroyBuffer(buffer: GPUBuffer) {
        const label = buffer.label;
        buffer.destroy();
        delete this._alloced_buffers[label];
    }
    logBuffers() {
        console.log("allocated buffers", this._alloced_buffers);
    }
}
