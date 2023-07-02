
import { Tensor } from "../torch/tensor";
import * as ops from "../torch/ops";
import * as nn from  "../torch/nn"
import * as factories from "../torch/factories";

// Tester

type logconfig = "always" | "fail" | "never";
interface test {
    "message": string,
    "func": string,
    "args": any,
    "target": any,
    "duration": number,
    "log": logconfig,
    "log_config": logconfig,
}

interface test_result { res: boolean, output: any, duration: number, msg?: string }

let express_duration: ("compare" | "difference" | "percent") = "difference"
const express_duration_func = (ts: number, torch: number): string => {
    if(express_duration == "compare") return `took ${ts}ms (${torch}ms in torch)`;
    if(express_duration == "difference") return `${ts-torch}ms slower`;
    if(express_duration == "percent") return `${ts/torch}x time to compute`;
    return `${ts}ms`
}  

let test_num;

async function run_test(func: string, args: any, target: any, control_duration: number, message?: string, log: logconfig="always", log_args: logconfig="never") {
    let { res, output, duration, msg } = await funcs[func](args, target);
    if(typeof(msg) == 'undefined') msg = "";
    if(log == "always" || (!res && log == "never")) {
        console.log(`🛠️ Running test #${test_num}: ${message}...`)
        if(log_args == "always" || (!res && log_args == "fail")) {
            console.log("Arguments: ", args);
        }
        if(log_args == "always" || (!res && log_args == "fail")) console.warn("Output: ", output, "\nTarget: ", target);
        if(res) {
            console.log(`✅ Passed! ${msg} `, express_duration_func(duration, control_duration));
        } else {
            console.warn(`🚩 Failed: ${msg} `, express_duration_func(duration, control_duration));
        }
    }
    return;
}

export async function run_tests(tests: Array<test>) {

    const indecies = ops.find_index(factories.ones([10, 10]));
    console.log("indecies: ", await indecies.toArrayAsync());

    for(test_num = 0; test_num < tests.length; test_num++) {
        await run_test(
            tests[test_num].func,
            tests[test_num].args,
            tests[test_num].target,
            tests[test_num].duration,
            tests[test_num].message,
            tests[test_num].log,
            tests[test_num].log_config,
        );
    }
}

// Helpers
function array_eq(a: Array<any>, b: Array<any>): number {
    if(a.length != b.length) return Infinity;
    const diff = (a.reduce((acc, v, i) => {
        return acc + Math.abs(v - b[i]);
    }, 0) / a.length);
    if(isNaN(diff)) return Infinity;
    return diff;
}

//
const funcs: { [key: string]: (args: any, target: any) => Promise<test_result>} = {
    "unsqueeze": test_unsqueeze,
    "squeeze": test_squeeze,
    "linear": test_linear,
    "nn_linear": test_nn_linear,
    "mm": test_mm,
    "transpose": test_transpose,
    "linspace": test_linspace,
    "conv2d": test_conv2d,
    "max_pool2d": test_max_pool2d,
    "scaled_dot_product_attention": test_scaled_dot_product_attention,
    "sum": test_sum,
    "scalar_add": test_scalar_add,
    "scalar_sub": test_scalar_sub,
    "scalar_mul": test_scalar_mul,
    "scalar_div": test_scalar_div,
    "group_norm": test_group_norm,
    "layer_norm": test_layer_norm,
    "chunk": test_chunk,
    "clamp": test_clamp,
    "silu": test_silu,
    "gelu": test_gelu,
    "softmax": test_softmax,
    "upsample": test_upsample,
    "cat": test_cat
}

// Tests

// WIP
async function test_nn_multihead_attention(args, target): Promise<test_result> {
    /**
     * args: {
     *  query: Tensor,
     *  key: Tensor,
     *  value: Tensor,
     *  embed_dim: number,
     *  num_heads: number,
     * }
    **/
   const mha = new nn.MultiheadAttention(args.emb_dim, args.num_heads);
   const query = ops.tensor(args.input);
   const key = ops.tensor(args.key);
   const value = ops.tensor(args.value);
   const target_output = ops.tensor(target);
   const start = Date.now();
   const actual_output = mha.forward(query, key, value);
   const duration = Date.now() - start;

   const output_data = await actual_output.output.toArrayAsync();

   if(array_eq(actual_output.output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
   if(array_eq(output_data.flat(4), target.flat(4)) > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

   return { res: true, output: output_data, duration: duration }
}

async function test_nn_layernorm(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  norm_shape: Shape
     * }
    **/

    const ln = new nn.LayerNorm(args.norm_shape);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
}

async function test_nn_groupnorm(args, target): Promise<test_result> {

}

async function test_nn_linear(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  inChannels: Tensor,
     *  outChannels: Tensor
     * }
    **/
    if(args.bias) console.warn("🟨 bias not yet implemented in test: \"nn_linear\"");
    const ln = new nn.Linear(args.inChannels, args.outChannels);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ln.forward(input);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(array_eq(output_data.flat(4), target.flat(4)) > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration }
}

async function test_conv2d(args, target): Promise<test_result> {

}

async function test_maxpool2d(args, target): Promise<test_result> {

}



async function test_cat(args, target): Promise<test_result> {
    /**
     * args: {
     *  a: Tensor,
     *  b: Tensor,
     *  dim: number
     * }
     */
    const a = ops.tensor(args.a);
    const b = ops.tensor(args.b);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.cat(a, b, args.dim);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_upsample(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  scale_factor: Shape,
     *  size: Shape,
     *  mode: string
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.upsample(input, args.size, args.scale_factor, args.mode);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_softmax(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  dim: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.softmax(input, args.dim);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    const sum = ops.sum(actual_output);
    const sum_target = ops.sum(target_output);
    console.log("softmax sum: ", await sum.toArrayAsync(), " actual sum: ", await sum_target.toArrayAsync());

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_silu(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.silu(input);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_gelu(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.gelu(input);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.0001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_clamp(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  low: number,
     *  high: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.clamp(input, args.low, args.high);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_chunk(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  chunks: number,
     *  dim: number
     * }
    **/
   const input = ops.tensor(args.input);
   const target_output = target.map((v) => { return ops.tensor(v); });
   const start = Date.now();
   const actual_output = ops.chunk(input, args.chunks, args.dim);
   const duration = Date.now() - start;
   const output_data = [];
   for(let i = 0; i < actual_output.length; i++) {
    output_data.push(await actual_output[i].toArrayAsync());
   }

   if(target_output.length != actual_output.length) return { res: false, output: output_data, duration: duration, msg: `mismatched number of chunks-- expected ${target_output.length}, got ${actual_output.length}` };
   for(let i = 0; i < target_output.length; i++) {
    if(array_eq(actual_output[i].shape, target_output[i].shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output[i].shape}, got ${actual_output[i].shape}` };
    if(array_eq(output_data[i].flat(4), target[i].flat(4)) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };
   }

    return { res: true, output: output_data, duration: duration };
}

async function test_group_norm(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     *  groups: number,
     *  weight: Tensor,
     *  bias: Tensor
     * }
     */
    const input = ops.tensor(args.input);
    const weight = ops.tensor(args.weight);
    const bias = ops.tensor(args.bias);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.group_norm(input, args.groups, weight, bias);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_layer_norm(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     *  norm_shape: number,
     *  weight: Tensor,
     *  bias: Tensor
     * }
     */
    const input = ops.tensor(args.input);
    const weight = ops.tensor(args.weight);
    const bias = ops.tensor(args.bias);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.layernorm(input, args.norm_shape, weight, bias);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_scalar_add(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     *  alpha: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.scalar_add(input, args.alpha);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_scalar_sub(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     *  alpha: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.scalar_sub(input, args.alpha);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_scalar_mul(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     *  alpha: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.scalar_mul(input, args.alpha);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_scalar_div(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     *  alpha: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.scalar_div(input, args.alpha);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_sum(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tesnor,
     * }
     */
    const input = ops.tensor(args.input);
    //const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = input.sum();
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };

}

async function test_scaled_dot_product_attention(args, target): Promise<test_result> {
    /**
     * args: {
     *  query: Tesnor,
     *  key: Tensor,
     *  value: Tensor,
     * }
     */
    const query = ops.tensor(args.query);
    const key = ops.tensor(args.key);
    const value = ops.tensor(args.value);

    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.scaled_dot_product_attention(query, key, value);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.01) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_max_pool2d(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  kernelSize: number,
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.maxpool2d(input, [args.kernelSize, args.kernelSize], [args.kernelSize, args.kernelSize], [0, 0], [1, 1]);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };

}

async function test_conv2d(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  weight: Tesnor,
     *  bias?: Tensor,
     * }
     */
    const input = ops.tensor(args.input);
    const weight = ops.tensor(args.weight);
    const bias = args.bias ? ops.tensor(args.bias) : undefined;
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.conv2d(input, weight, bias);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();
    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_linspace(args, target): Promise<test_result> {
    /**
     * args: {
     *  start: number,
     *  end: number,
     *  steps: number,
     * }
    **/
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = factories.linspace(args.start, args.end, args.steps);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();
    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_transpose(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     * }
    **/
    const input = ops.tensor(args.input)
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = input.transpose(args.dim0, args.dim1);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();
    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(array_eq(output_data.flat(4), target.flat(4)) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration };
}

async function test_mm(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  weights: Tensor,
     * }
    **/
    const input = ops.tensor(args.input);
    const weight = ops.tensor(args.weight);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = input.mm(weight);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();
    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
}

async function test_linear(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  weights: Tensor,
     *  bias: Tensor
     * }
    **/
    const input = ops.tensor(args.input);
    const weight = ops.tensor(args.weight);
    const bias =  args.bias != null ? ops.tensor(args.bias) : null;
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.linear(input, weight, bias);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();
    
    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(array_eq(output_data.flat(4), target.flat(4)) > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

   return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
}

async function test_unsqueeze(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Array,
     *  dim: number
     * }
    **/
    const input = ops.tensor(args.input);
    const target_ouput = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.unsqueeze(input, args.dim);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_ouput.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_ouput.shape}, got ${actual_output.shape}` };
    if(array_eq(output_data.flat(4), target.flat(4)) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration }
}

async function test_squeeze(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Array,
     *  dim: number
     * }
    **/
    const input = ops.tensor(args.input);
    const target_tensor = ops.tensor(target);
    const start = Date.now();
    const output = ops.squeeze(input, args.dim);
    const duration = Date.now() - start;
    const output_data = await output.toArrayAsync();

    if(array_eq(output.shape, target_tensor.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_tensor.shape}, got ${output.shape}` };
    if(array_eq(output_data.flat(4), target.flat(4)) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content` };

    return { res: true, output: output_data, duration: duration }
}

export default run_tests;