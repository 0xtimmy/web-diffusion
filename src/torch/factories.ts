import { Shapeish, defaultStrides, getShape } from "./shape";
import { Deviceish } from "./device";
import { getDevice } from "./devices";
import { Dtype, getDtype } from "./dtype";
import { Tensor } from "./tensor";

export function ones(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    return constant(shape, 1);
}

export function zeros(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    return constant(shape, 0);
}

export function empty(
    shape: Shapeish,
    dtype?: Dtype,
    device?: Deviceish 
): Tensor {
    const d = getDevice(device);
    const s = getShape(shape);
    const dt = getDtype(dtype);
    const storage = d.allocFor(s, dt);
    return new Tensor({
        data: storage,
        dtype: dt,
        shape: s,
        strides: defaultStrides(s),
        device: d,
    });
}

export function constant(
    shape: Shapeish,
    val=0,
    dtype?: Dtype,
    device?: Deviceish
): Tensor {
    const d = getDevice(device);
    const s = getShape(shape);
    const dt = getDtype(dtype);
    const storage = d.allocFor(s, dt);
    const array = storage.getTypedArray(dt);
    array.fill(val);
    return new Tensor({
        data: storage,
        dtype: dt,
        shape: s,
        strides: defaultStrides(s),
        device: d,
    });
}

export function uniform(
    shape: Shapeish,
    min= 0,
    max=1,
    dtype?: Dtype,
    device?: Deviceish 
): Tensor {
    const d = getDevice(device);
    const s = getShape(shape);
    const dt = getDtype(dtype);
    const storage = d.allocFor(s, dt);
    const array = storage.getTypedArray(dt);
    array.fill(0);
    const diff = max - min;
    for (let i = 0; i < array.length; i++) {
        array[i] = Math.random() * diff + min;
    }
    return new Tensor({
        data: storage,
        dtype: dt,
        shape: s,
        strides: defaultStrides(s),
        device: d,
    });
}

export function normal(
    shape: Shapeish,
    mean=0,
    std=1,
    dtype?: Dtype,
    device?: Deviceish 
): Tensor {
    return uniform(shape, 0, 1, dtype, device).box_muller(mean, std);
}
