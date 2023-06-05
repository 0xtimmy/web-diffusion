import * as torch from "@/torch"

export function linspace(start: number, end: number, steps: number): torch.Tensor {
    const out = new Float32Array(steps)
    const diff = (end - start)  / steps;
    for(let i = 0; i < out.length; i++) {
        out[i] = diff * i + start;
    }
    return torch.tensor(Array.from(out), "float32");
}

export async function cumprod(input: torch.Tensor, dim=0): Promise<torch.Tensor> {
    const out = Array.from(await input.toArrayAsync()) as Array<number>;
    for(let i = 1; i < out.length; i++) {
        out[i] = out[i] * out[i - 1];
    }
    return torch.tensor(out, "float32");
}

export function randint(low: number, high: number, size: number): torch.Tensor {
    const out = new Array(size);
    const diff = high - low;
    for(let i = 0; i < out.length; i++) {
        out[i] = Math.random() * diff + low;
    }
    return torch.tensor(out, "uint32");
}

export function randn(shape: torch.Shape): torch.Tensor {
    const _randn = (_shape: torch.Shape) => {
        const out = new Array(_shape[0]);
        for(let i = 0; i < _shape[0]; i++) {
            if(_shape.length <= 1) {
                out[i] = Math.random();
            } else {
                out[i] = _randn(Array.from(_shape).splice(1));
            }
        }
        return out;
    }
    const res = _randn(shape);
    console.log(res);
    return torch.tensor(res);
}

export function clamp(input: torch.Tensor, low: number, high: number): torch.Tensor {
    return torch.max(torch.min(input, high), low);
}

export function arange(start: number, end: number, step: number): torch.Tensor {
    const out = [];
    for(let i = start; i < end; i += step) {
        out.push(i)
    }
    return torch.tensor(out);
}

export function repeat(t: torch.Tensor, shape: torch.Shape): torch.Tensor {
    return torch.cat(t, t, 0);
}

/*
export function cat(a: torch.Tensor, b: torch.Tensor, dim: number): torch.Tensor {
    console.log("cat a: ", a, " and b: ", b);

    return a;
}
*/
