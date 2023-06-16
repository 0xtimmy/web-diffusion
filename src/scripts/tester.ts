
import { Tensor } from "../torch/tensor";
import * as aops from "../torch/ops_artisanal";

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

let test_num;

async function run_test(func: string, args: any, target: any, message?: string, log: logconfig="always", log_args: logconfig="never") {
    const { res, output, error_msg } = await funcs[func](args, target);
    if(log == "always" || (!res && log == "never")) {
        console.log(`🛠️ Running test #${test_num}: ${message}...`)
        if(log_args == "always" || (!res && log_args == "never")) {
            console.log("Arguments: ", args);
        }
        if(log_args == "always" || (!res && log_args == "never")) console.warn("Output: ", output);
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
}

// Tests

async function test_unsqueeze(args, target): Promise<{ res: boolean, output: any, error_msg?: string }> {
    /**
     * args: {
     *  input: Array,
     *  dim: number
     * }
    **/
    const input = aops.tensor(args.input);
    const target_tensor = aops.tensor(target);
    const output = aops.unsqueeze(input, args.dim)
    const output_data = await output.toArrayAsync();

    if(!array_eq(output.shape, target_tensor.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_tensor.shape}, got ${output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data }
}

async function test_squeeze(args, target): Promise<{ res: boolean, output: any, error_msg?: string }> {
    /**
     * args: {
     *  input: Array,
     *  dim: number
     * }
    **/
    const input = aops.tensor(args.input);
    const target_tensor = aops.tensor(target);
    const output = aops.squeeze(input, args.dim)
    const output_data = await output.toArrayAsync();

    if(!array_eq(output.shape, target_tensor.shape)) return { res: false, output: output_data, error_msg: `mismatched shapes-- expected ${target_tensor.shape}, got ${output.shape}` };
    if(!array_eq(output_data.flat(4), target.flat(4))) return { res: false, output: output_data, error_msg: `mismatched tensor content` };

    return { res: true, output: output_data }
}

export default run_tests;