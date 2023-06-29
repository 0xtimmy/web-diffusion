import { Shape, shapeSize } from "./shape";
import { ATypedArray, Dtype, dtypeByteSize } from "./dtype";
import type { UntypedStorage } from "./storage";
import {
    Kernel,
    KernelConfig,
    KernelConfigInput,
    KernelKey,
    KernelSpec,
    getKernelConfig,
    getKernelKey,
} from "./kernel";
import { registry as kernelRegistry } from "./kernels";

export type DeviceType = "cpu" | "webgpu";
export type DeviceId = string;

export type Deviceish = DeviceType | Device | DeviceId;

export abstract class Device {
    private _id: DeviceId;
    private _type: DeviceType;
    private _kernels: { [key: string]: Kernel } = {};
    get id(): DeviceId {
        return this._id;
    }
    get type(): DeviceType {
        return this._type;
    }
    constructor(id: DeviceId, type: DeviceType) {
        this._id = id;
        this._type = type;
    }
    abstract alloc(byteSize: number): UntypedStorage;
    allocFor(shape: Shape, dtype: Dtype): UntypedStorage {
        const elementByteSize = dtypeByteSize(dtype);
        const byteSize = shapeSize(shape) * elementByteSize;
        return this.alloc(byteSize);
    }
    getKernel(name: string, config: KernelConfigInput, params=null): Kernel {
        const spec = kernelRegistry[name];
        if (spec === undefined) {
            throw new Error(`Kernel "${name}" not found`);
        }
        const kconfig = getKernelConfig(spec, config);
        // Time to create kernel seemed negligible;
        /*
        const key = getKernelKey(spec, kconfig);
        let kernel = this._kernels[key];
        if (kernel === undefined) {
            //const startCreateKernel = Date.now();
            kernel = this.createKernel(spec, kconfig);
            //const doneCreatingKernel = Date.now();
            //console.log("time to create kernel: ", startCreateKernel - doneCreatingKernel, "ms");
            this._kernels[key] = kernel;
        }
        */
        const kernel = this.createKernel(spec, kconfig, params);
        return kernel;
    }
    abstract createKernel(spec: KernelSpec, config: KernelConfig, params?: null | any): Kernel;
    abstract getBufferForKernel(
        storage: UntypedStorage,
        dtype: Dtype
    ): ATypedArray | GPUBuffer;
    abstract getStorageFromKernel(
        buffer: ATypedArray | GPUBuffer
    ): UntypedStorage;
}
