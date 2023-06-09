import { Tensor } from "./tensor";

function _calculate_fan_in_fan_out(tensor: Tensor) {
    const dimensions = tensor.dim;
    if (dimensions < 2) throw new Error("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions");

    const num_input_fmaps = tensor.shape[1];
    const num_output_fmaps = tensor.shape[0];
    let receptive_field_size = 1;
    if(tensor.dim > 2) {
        for (let i = 2; i < tensor.shape.length; i++) {
            receptive_field_size *= tensor.shape[i];
        }
    }
    const fan_in = num_input_fmaps * receptive_field_size;
    const fan_out = num_output_fmaps * receptive_field_size;
    return { fan_in, fan_out };

}

function _no_grad_uniform(tensor: Tensor, a: number, b: number) {
    return tensor.uniform(a, b);
}

export function xavier_uniform(input: Tensor, gain=1.0) {
    const { fan_in, fan_out } = _calculate_fan_in_fan_out(input);
    const std = gain * Math.sqrt(2.0 / (fan_in + fan_out));
    const a = Math.sqrt(3.0) * std;
    return input.uniform(-a, a);
}

export function xavier_normal(input: Tensor, gain=1.0) {
    const { fan_in, fan_out } = _calculate_fan_in_fan_out(input)
    const std = gain * Math.sqrt(2.0 / (fan_in + fan_out))

    return input.normal(0.0, std);
}

