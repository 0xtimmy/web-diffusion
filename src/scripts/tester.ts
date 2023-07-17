
import * as ops from "../torch/ops";
import * as nn from  "../torch/nn"
import * as factories from "../torch/factories";
import { Tensor } from "../torch/tensor"
import { 
    SelfAttention,
    DoubleConv,
    Down,
    Up,
    UNet
} from "@/components/diffuser/modules"
import { Diffusion } from "@/components/diffuser/ddpm";

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

async function run_test(func: string, args: any, target: any, control_duration: number, message?: string, log: logconfig="always", log_args: logconfig="never", handleStep?: any): Promise<Array<test_result>> {
    let results = await funcs[func](args, target, handleStep);
    if(typeof((results as any).length) == 'undefined') results = [results as any];
    return (results as Array<test_result>).map((result: test_result) => {
        let { res, output, duration, msg } = result;
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
    })
    
}

export async function run_tests(tests: Array<test>, handleStep: any) {

    const results = [];
    let avg_percent_diff = 0; // ts / python
    for(test_num = 0; test_num < tests.length; test_num++) {
        (await run_test(
            tests[test_num].func,
            tests[test_num].args,
            tests[test_num].target,
            tests[test_num].duration,
            tests[test_num].message,
            tests[test_num].log,
            tests[test_num].log_config,
            handleStep,
        )).forEach((result, i) => {
            results.push({
                test: `${tests[test_num]}.${i}`,
                result: result
            });
            avg_percent_diff += (results[test_num].result.duration / (results[test_num].test.duration + 0.00001)) / tests.length;
        })
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
const funcs: { [key: string]: (args: any, target: any, handleStep?: any) => Promise<test_result | Array<test_result>> } = {
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
    "permute": test_permute,
    "unet": test_unet,
    "pos_enc": test_pos_enc,
    "ddpm": test_ddpm,
    "randn": test_randn,
    "uniform": test_uniform,
    "repeat": test_repeat,
}

// Tests

async function test_repeat(args, target): Promise<test_result> {

    const input = ops.tensor(args.input);
    const start = Date.now();
    const actual_output = ops.repeat(input, args.repeat);
    const duration = Date.now() - start;

    // old repeat
    //const _start = Date.now();
    //const _actual_output = ops._repeat(input, args.repeat);
    //const _duration = Date.now() -_start;
    //console.log(`new repeat took: ${duration / _duration} the time to complete`)

    const target_output = ops.tensor(target);
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };
    
    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_randn(args, target): Promise<test_result> {
    const start = Date.now();
    const actual_output = factories.randn([args.size]);
    const duration = Date.now() - start;

    const output_data = (await actual_output.toArrayAsync()).flat(4);
    target = target.flat(4);

    console.log("randn: ", output_data);
    const actual_mean = (output_data as Array<number>).reduce((acc, v) => {
        return acc + v / output_data.length;
    }, 0)
    const actual_variance = (output_data as Array<number>).reduce((acc, v) => {
        return acc + Math.pow((v - actual_mean), 2) / output_data.length;
    })
    const actual_std = Math.sqrt(Math.abs(actual_variance));

    const target_mean = (target as Array<number>).reduce((acc, v) => {
        return acc + v / target.length;
    }, 0)
    const target_variance = (target as Array<number>).reduce((acc, v) => {
        return acc + Math.pow((v - target_mean), 2) / target.length;
    })
    const target_std = Math.sqrt(Math.abs(target_variance));

    if(
        Math.abs(actual_mean - target_mean) > 0.1 ||
        Math.abs(actual_variance - target_variance) > 0.1 ||
        Math.abs(actual_std - target_std) > 0.1
    ) return { res: false, output: output_data, duration: duration, msg: `mismatched stats, target mean=${target_mean}, target var=${target_variance}, target std=${target_std}; actual mean=${actual_mean}, actual var=${actual_variance}, actual std=${actual_std}` };
    
    return { res: true, output: output_data, duration: duration, msg: `target mean=${target_mean}, target var=${target_variance}, target std=${target_std}; actual mean=${actual_mean}, actual var=${actual_variance}, actual std=${actual_std}` };
}

async function test_uniform(args, target): Promise<test_result> {
    const start = Date.now();
    const actual_output = factories.uniform([args.size], 0, 1);
    const duration = Date.now() - start;

    const output_data = await actual_output.toArrayAsync();
    const target_output = ops.tensor(target);

    console.log("uniform: ", output_data);

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };
    
    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };

}

async function test_ddpm(args, target, handleStep?: any): Promise<test_result> {

    const input = ops.tensor(args.input);

    const model = new UNet();
    await model.loadStateDictFromURL("../../parameters/pokemon");
    const diffuser = new Diffusion({ noise_steps: args.noise_steps, img_size: 64 });

    
    const start = Date.now();
    const actual_output = await diffuser._sample(model, input, args.noises, args.results, handleStep);
    const duration = Date.now() - start;

    const target_output = ops.tensor(target);
    
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };
    
    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_pos_enc(args, target): Promise<test_result> {
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);

    const start = Date.now();
    
    const range = factories.arange(0, args.channels, 2).scalar_div(args.channels);
    const inv_freq = factories.constant(range.shape, 10000).pow(range).scalar_pow(-1);
    const pos_enc_a = ops.repeat(input, [1, Math.floor(args.channels / 2)]).mul(inv_freq).sin();
    const pos_enc_b = ops.repeat(input, [1, Math.floor(args.channels / 2)]).mul(inv_freq).cos();
    const actual_output = ops.cat(pos_enc_a, pos_enc_b, 1);

    const duration = Date.now() - start;
    const output_data = await actual_output.toArrayAsync();

    if(array_eq(actual_output.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    const diff = array_eq(output_data.flat(4), target.flat(4));
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

    return { res: true, output: output_data, duration: duration, msg: `average diff: ${diff}` };
}

async function test_unet(args, target): Promise<Array<test_result>> {

    const model = new UNet();
    await model.loadStateDictFromURL("../../parameters/pokemon");

    const check = async (key: string, a: Tensor, target: any): Promise<test_result> => {

        const output_data = await a.toArrayAsync();
        const target_output = ops.tensor(target);

        if(array_eq(a.shape, target_output.shape) > 0) return { res: false, output: output_data, duration: duration, msg: `${key}: mismatched shapes-- expected ${target_output.shape}, got ${a.shape}` };
        const diff = array_eq(output_data.flat(4), target.flat(4));
        if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `${key}: mismatched tensor content, average diff: ${diff}` };

        return { res: true, output: output_data, duration: duration, msg: `${key}: average diff: ${diff}` };
    }

    const input = ops.tensor(args.input);
    let t = ops.tensor(args.t);

    const start = Date.now();

    t = ops.unsqueeze(t, -1);
    const pos_enc = model.pos_encoding(t, model.time_dim);
    
    const inc = model.inc.forward(input);
    const down1 = model.down1.forward(inc, pos_enc);
    const sa1 =  model.sa1.forward(down1);
    const down2 =  model.down2.forward(sa1, pos_enc);
    const sa2 =  model.sa2.forward(down2);
    const down3 =  model.down3.forward(sa2, pos_enc);
    const sa3 =  model.sa3.forward(down3);

    const bot1 =  model.bot1.forward(sa3);
    const bot2 =  model.bot2.forward(bot1);
    const bot3 =  model.bot3.forward(bot2);

    const up1 = model.up1.forward(bot3, sa2, pos_enc);
    const sa4 =  model.sa4.forward(up1);
    const up2 =  model.up2.forward(sa4, sa1, pos_enc);
    const sa5 =  model.sa5.forward(up2);
    const up3 = model.up3.forward(sa5, inc, pos_enc);
    const sa6 =  model.sa6.forward(up3);
    
    const output = model.outc.forward(sa6);

    const duration = Date.now() - start;

    return [
        await check("pos_enc", pos_enc, args.pos_enc),
        await check("inc", inc, args.inc),
        await check("down1", down1, args.down1),
        await check("sa1", sa1, args.sa1),
        await check("down1", down2, args.down2),
        await check("sa2", sa2, args.sa2),
        await check("down3", down3, args.down3),
        await check("sa3", sa3, args.sa3),

        await check("bot1", bot1, args.bot1),
        await check("bot2", bot2, args.bot2),
        await check("bot3", bot3, args.bot3),

        await check("up1", up1, args.up1),
        await check("sa4", sa4, args.sa4),
        await check("up2", up2, args.up2),
        await check("sa5", sa5, args.sa5),
        await check("up3", up3, args.up3),
        await check("sa6", sa6, args.sa6),

        await check("output", output, target),
    ]
    
}

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

    alpha = ops.index(alpha, t)//.repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
    alpha_hat = ops.index(alpha_hat, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
    beta = ops.index(beta, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);

    const start = Date.now()
    
    let one_div_sqrt_alpha = ops.sqrt(alpha).scalar_pow(-1);
    
    let sqrt_one_minus_alpha_hat = alpha_hat.scalar_mul(-1).scalar_add(1).sqrt();
    let one_minus_alpha = alpha.scalar_mul(-1).scalar_add(1);
    const alpha_div_alpha_hat = ops.div(one_minus_alpha, sqrt_one_minus_alpha_hat);
    one_minus_alpha.destroy();
    sqrt_one_minus_alpha_hat.destroy();
    let predicted_noise = input.mul(alpha_div_alpha_hat);
    alpha_div_alpha_hat.destroy();
    let nx = x.sub(predicted_noise);
    
    nx = nx.mul(one_div_sqrt_alpha);
    one_div_sqrt_alpha.destroy();
    
    const beta_noise = beta.sqrt().mul(noise);
    noise.destroy();
    
    const actual_output = nx.add(beta_noise);
    beta_noise.destroy();
    
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
    ln.loadStateDict(args.state_dict);
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
     *  kernel_size: number,
     *  padding: number,
     *  state_dict: StateDict
     * }
    **/
    const conv = new nn.Conv2d(args.in_channels, args.out_channels, args.kernel_size, 1, args.padding);
    conv.loadStateDict(args.state_dict);
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
    if(diff > 0.00001) return { res: false, output: output_data, duration: duration, msg: `mismatched tensor content, average diff: ${diff}` };

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