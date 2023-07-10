
import * as ops from "../torch/ops";
import * as nn from  "../torch/nn"
import * as factories from "../torch/factories";
import { 
    SelfAttention,
    DoubleConv,
    Down,
    Up
} from "@/components/diffuser/modules"

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
    return { res, output, duration, msg };
}

export async function run_tests(tests: Array<test>) {

    const results = [];
    let avg_percent_diff = 0; // ts / python
    for(test_num = 0; test_num < tests.length; test_num++) {
        results.push({
            test: tests[test_num],
            result: await run_test(
                tests[test_num].func,
                tests[test_num].args,
                tests[test_num].target,
                tests[test_num].duration,
                tests[test_num].message,
                tests[test_num].log,
                tests[test_num].log_config,
            )
        });
        avg_percent_diff += (results[test_num].result.duration / (results[test_num].test.duration + 0.00001)) / tests.length;
    }

    console.log(`🏁 Testing complete, completed in ${avg_percent_diff}% time as in pytorch`);
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
    "cat": test_cat,
    "nn_multihead_attention": test_nn_multihead_attention,
    "nn_layernorm": test_nn_layernorm,
    "nn_groupnorm": test_nn_groupnorm,
    "nn_linear": test_nn_linear,
    "nn_conv2d": test_nn_conv2d,
    "nn_maxpool2d": test_nn_maxpool2d,
    "linear_model_loading": test_linear_model_loading,
    "compound_model_loading": test_compound_model_loading,
    "cumprod": test_cumprod,
    "self_attention": test_self_attention,
    "double_conv": test_double_conv,
    "down": test_down,
    "up": test_up,
    "denoise": test_denoise,
    "permute": test_permute
}

// Tests

async function test_permute(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  permute: Array<number>
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);

    const start = Date.now();
    const actual_output = input.permute(args.permute);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_denoise(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  i: number
     *  original_noise: Tensor,
     *  noise: Tensor,
     *  alpha: Tensor,
     *  alpha_hat: Tensor,
     *  beta: Tensor
     * }
    **/
    
    let x = ops.tensor(args.original_noise);
    const input = ops.tensor(args.input);
    
    let beta = factories.linspace(1e-4, 0.02, 1000);
    let alpha = ops.sub(factories.ones([1000]), beta);
    let alpha_hat = ops.cumprod(alpha);

    let noise = ops.tensor(args.noise);

    const t = factories.constant([1], args.i);
    console.log(await t.toArrayAsync(), await alpha_hat.toArrayAsync());

    alpha = alpha.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
    alpha_hat = alpha_hat.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
    beta = beta.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);

    const start = Date.now()

    let one_div_sqrt_alpha = ops.div(factories.ones(alpha.shape), ops.sqrt(alpha));
    let sqrt_one_minus_alpha_hat = ops.sqrt(ops.sub(factories.ones(alpha_hat.shape), alpha_hat));
    let one_minus_alpha = ops.sub(factories.ones(alpha.shape), alpha);
    let predicted_noise = ops.mul(input, ops.div(one_minus_alpha, sqrt_one_minus_alpha_hat));
    x = ops.sub(x, predicted_noise);
    x = ops.mul(one_div_sqrt_alpha, x);
    const beta_noise = ops.mul(ops.sqrt(beta), noise);
    const actual_output = ops.add(x, beta_noise);
    console.log(await t.toArrayAsync(), await alpha_hat.toArrayAsync());
    
    const duration = Date.now() - start;

    const target_output = ops.tensor(target)
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_down(args, target): Promise<test_result> {
    /**
     * args: {
     *  in_channels: number,
     *  out_channels: number,
     *  input: Tensor,
     *  t: Tensor,
     *  state_dict: { weight: Tensor, bias: Tensor }
     * }
     */
    const down = new Down(args.in_channels, args.out_channels);
    const start = Date.now()
    down.loadStateDict(args.state_dict);
    const duration = Date.now() - start;

    const input = ops.tensor(args.input);
    const t = ops.tensor(args.t);
    const target_output = ops.tensor(target);
    const actual_output = await down.forward(input, t);

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_up(args, target): Promise<test_result> {
    /**
     * args: {
     *  in_channels: number,
     *  out_channels: number,
     *  input: Tensor,
     *  skip: Tensor
     *  t: Tensor,
     *  state_dict: { weight: Tensor, bias: Tensor }
     * }
     */
    const up = new Up(args.in_channels, args.out_channels);
    const start = Date.now()
    up.loadStateDict(args.state_dict);
    const duration = Date.now() - start;

    const input = ops.tensor(args.input);
    const t = ops.tensor(args.t);
    const skip = ops.tensor(args.skip);
    const target_output = ops.tensor(target);
    const actual_output = up.forward(input, skip, t);

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_double_conv(args, target): Promise<test_result> {
    /**
     * args: {
     *  in_channels: number,
     *  out_channels: number,
     *  mid_channels: number,
     *  residual: boolean
     *  input: Tensor,
     *  state_dict: { weight: Tensor, bias: Tensor }
     * }
     */
    const conv = new DoubleConv(args.in_channels, args.out_channels, args.mid_channels, args.residual);
    const start = Date.now()
    conv.loadStateDict(args.state_dict);
    const duration = Date.now() - start;

    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const actual_output = await conv.forward(input);

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_self_attention(args, target): Promise<test_result> {
    /**
     * args: {
     *  channels: number,
     *  size: number,
     *  input: Tensor,
     *  state_dict: { weight: Tensor, bias: Tensor }
     * }
     */
    const sa = new SelfAttention(args.channels, args.size);
    const start = Date.now()
    sa.loadStateDict(args.state_dict);
    const duration = Date.now() - start;

    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const actual_output = await sa.forward(input);

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_cumprod(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  dim: number
     * }
     */
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ops.cumprod(input);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync()

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4))
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_linear_model_loading(args, target): Promise<test_result> {
    /**
     * args: {
     *  in_channels: number,
     *  out_channels: number,
     *  input: Tensor,
     *  state_dict: { weight: Tensor, bias: Tensor }
     * }
     */
    const ln = new nn.Linear(args.in_channels, args.out_channels);
    const start = Date.now()
    ln.loadStateDict(args.state_dict);
    const duration = Date.now() - start;

    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const actual_output = ln.forward(input);

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_compound_model_loading(args, target): Promise<test_result> {
    /**
     * args: {
     *  in_channels: number,
     *  mid_channels: number,
     *  out_channels: number,
     *  input: Tensor,
     *  state_dict: { weight: Tensor, bias: Tensor }
     * }
     */
    const model = new nn.Sequential([
        new nn.Linear(args.in_channels, args.mid_channels),
        new nn.Linear(args.mid_channels, args.out_channels)
    ]);

    const start = Date.now()
    model.loadStateDict(args.state_dict);
    const duration = Date.now() - start;

    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const actual_output = await model.forward(input);

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_nn_multihead_attention(args, target): Promise<test_result> {
    /**
     * args: {
     *  query: Tensor,
     *  key: Tensor,
     *  value: Tensor,
     *  embed_dim: number,
     *  num_heads: number,
     *  state_dict: StateDict
     * }
    **/
   const mha = new nn.MultiheadAttention(args.embed_dim, args.num_heads);
   mha.loadStateDict(args.state_dict);

   const query = ops.tensor(args.query);
   const key = ops.tensor(args.key);
   const value = ops.tensor(args.value);

   const target_output = ops.tensor(target);

    //const raw = ops.scaled_dot_product_attention(query, key, value);
    //console.log("raw scaled dot product attention", await raw.toArrayAsync());

   const start = Date.now();
   const actual_output = mha.forward(query, key, value);
   const duration = Date.now() - start;

   const output_data = await actual_output.output.toArrayAsync();

   if(array_eq(actual_output.output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.output.shape}` };
   const diff = array_eq(output_data.flat(4), target.flat(4));
   if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

   return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
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
    const start = Date.now();
    const actual_output = ln.forward(input);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
}

async function test_nn_groupnorm(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  num_groups: number,
     *  num_channels: number,
     * }
    **/

    const gn = new nn.GroupNorm(args.num_groups, args.num_channels);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = gn.forward(input);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
}

async function test_nn_linear(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  in_channels: Tensor,
     *  out_channels: Tensor
     * }
    **/
    const ln = new nn.Linear(args.in_channels, args.out_channels);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = ln.forward(input);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
}

async function test_nn_conv2d(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  in_channels: number,
     *  out_channels: number,
     *  kernel_size: number
     * }
    **/
    const conv = new nn.Conv2d(args.in_channels, args.out_channels, args.kernel_size);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = conv.forward(input);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
}

async function test_nn_maxpool2d(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  kernel_size: number
     * }
    **/
    const pool = new nn.MaxPool2d(args.kernel_size);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const start = Date.now();
    const actual_output = pool.forward(input);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` }
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
    const actual_output = input.scalar_add(args.alpha);
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
    const actual_output = input.scalar_sub(args.alpha);
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
    const actual_output = input.scalar_mul(args.alpha);
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
    const actual_output = input.scalar_div(args.alpha);
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
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

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
    const actual_output = ops.conv2d(input, weight, bias, 1, 0, 1, 1);
    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();
    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), [target].flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

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
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

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
    const bias =  args.bias != null ? ops.tensor(args.bias) : undefined;
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