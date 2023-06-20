
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
    "log": logconfig,
    "log_config": logconfig,
}

interface test_result { res: boolean, output: any, error_msg?: string }

let test_num;

async function run_test(func: string, args: any, target: any, message?: string, log: logconfig="always", log_args: logconfig="never") {
    const { res, output, error_msg } = await funcs[func](args, target);
    if(log == "always" || (!res && log == "never")) {
        console.log(`🛠️ Running test #${test_num}: ${message}...`)
        if(log_args == "always" || (!res && log_args == "fail")) {
            console.log("Arguments: ", args);
        }
        if(log_args == "always" || (!res && log_args == "fail")) console.warn("Output: ", output, "\nTarget: ", target);
        if(res) {
            console.log(`✅ Passed!`);
        } else {
            console.warn(`🚩 Failed: ${error_msg}`);
        }
    }
    return;
}

export async function run_tests(tests: Array<test>) {
    for(test_num = 0; test_num < tests.length; test_num++) {
        await run_test(
            tests[test_num].func,
            tests[test_num].args,
            tests[test_num].target,
            tests[test_num].message,
            tests[test_num].log,
            tests[test_num].log_config,
        );
    }
}

// Helpers
function array_eq(a: Array<any>, b: Array<any>, tolerance=0) {
    if(a.length != b.length) return false;
    return (a.reduce((acc, v, i) => {
        return acc + Math.abs(v - b[i]);
    }, 0) / a.length) <= tolerance;
}

//
const funcs: { [key: string]: (args: any, target: any) => Promise<{ res: boolean, output: any, error_msg?: string }>} = {
    "unsqueeze": test_unsqueeze,
    "squeeze": test_squeeze,
    "linear": test_linear,
    "nn_linear": test_nn_linear,
    "mm": test_mm,
    "transpose": test_transpose,
    "linspace": test_linspace,
    "conv2d": test_conv2d
}

// Tests

// WIP
async function test_nn_linear(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     *  weights: Tensor,
     *  bias: Tensor
     * }
    **/
    if(args.bias) console.warn("🟨 bias not yet implemented in test: \"nn_linear\"");
    const ln = new nn.Linear(args.inChannels, args.outChannels);
    const input = ops.tensor(args.input);
    const target_output = ops.tensor(target);
    const actual_output = ln.forward(input);

    const output_data = await actual_output.toArrayAsync();

    if(!array_eq(actual_output.shape, target_output.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data }
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
    const actual_output = ops.conv2d(input, weight, bias);
    const output_data = await actual_output.toArrayAsync();
    if(!array_eq(actual_output.shape, target_output.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4), 0.0001)) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data };
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
    const actual_output = factories.linspace(args.start, args.end, args.steps);
    const output_data = await actual_output.toArrayAsync();
    if(!array_eq(actual_output.shape, target_output.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4), 0.0000001)) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data };
}

async function test_transpose(args, target): Promise<test_result> {
    /**
     * args: {
     *  input: Tensor,
     * }
    **/
    const input = ops.tensor(args.input)
    const target_output = ops.tensor(target)
    const actual_output = input.transpose(args.dim0, args.dim1);
    const output_data = await actual_output.toArrayAsync();
    if(!array_eq(actual_output.shape, target_output.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data };
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
    const actual_output = input.mm(weight);
    const output_data = await actual_output.toArrayAsync();
    if(!array_eq(actual_output.shape, target_output.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data }
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
    const actual_output = ops.linear(input, weight, bias);
    const output_data = await actual_output.toArrayAsync();
    if(!array_eq(actual_output.shape, target_output.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

   return { res: true, output: output_data }
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
    const actual_output = ops.unsqueeze(input, args.dim)
    const output_data = await actual_output.toArrayAsync();

    if(!array_eq(actual_output.shape, target_ouput.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_ouput.shape}, got ${actual_output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data }
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
    const output = ops.squeeze(input, args.dim)
    const output_data = await output.toArrayAsync();

    if(!array_eq(output.shape, target_tensor.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_tensor.shape}, got ${output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data }
}

export default run_tests;