import * as torch from "@/torch";
import {
    linspace,
    randint,
    randn,
    cumprod,
    clamp,
    repeat
} from "./utils"

export default async function() {
    await test_upsample();
    //await test_maxpool2d();
    //await test_layernorm();
    //await test_conv();
    //await test_repeat();
    //await test_cat();
    //await test_minmax();
    //await test_scalars();
    //await test_linspace();
    //await test_randint();
    //await test_randn();
    //await test_cumprod();
    //await test_clamp();
}

function array_eql(a: Array<any>, b: Array<any>) {
    if(a.length != b.length) {
        console.warn("Arrays must have the same length to be compared");
        return false;
    }
    let e = 0;
    let maxe = 0;
    for(let i = 0; i < a.length; i++) {
        const diff = Math.abs(a[i] - b[i]);
        e += diff;
        maxe = diff > maxe ? diff : maxe;
    }
    if(e == 0) return true;
    console.warn("max error = ", maxe, "; average error = ", e / a.length);
    return false;
}

async function check(res: torch.Tensor, expected: torch.Tensor): Promise<boolean> {
    const res_data      = await res.toArrayAsync();
    const actual_data   = await expected.toArrayAsync();
    if(!array_eql(res.shape, expected.shape)) {
        console.warn("‼️ FAILED - shapes don't match");
    } else {
        const match = array_eql(res_data.flat(4), actual_data.flat(4));
        if(match) {
            console.log("✅ PASSED");
        } else {
            console.warn("‼️ FAILED - content doesn't match");
        }
    }
    console.log("Got: ", res_data, "\nExpected: ", actual_data);
    return false;
    
}

const UPSAMPLE_INPUT        = torch.tensor([[[1, 2, 3, 4]]]);
const UPSAMPLE_RESULT       = torch.tensor([[[1, 1, 2, 2, 3, 3, 4, 4]]])
const UPSAMPLE_2D_INPUT     = torch.tensor([[[[1, 2], [3, 4]]]]);
const UPSAMPLE_2D_RESULT    = torch.tensor([[[[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4], [3, 3, 3, 4, 4, 4]]]]);
const UPSAMPLE_3D_INPUT     = torch.tensor([[[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]]]);
const UPSAMPLE_3D_RESULT    = torch.tensor([[[[[0.1000, 0.1000, 0.2000, 0.2000],
                                                [0.1000, 0.1000, 0.2000, 0.2000],
                                                [0.3000, 0.3000, 0.4000, 0.4000],
                                                [0.3000, 0.3000, 0.4000, 0.4000]],

                                            [[0.1000, 0.1000, 0.2000, 0.2000],
                                                [0.1000, 0.1000, 0.2000, 0.2000],
                                                [0.3000, 0.3000, 0.4000, 0.4000],
                                                [0.3000, 0.3000, 0.4000, 0.4000]],

                                            [[0.5000, 0.5000, 0.6000, 0.6000],
                                                [0.5000, 0.5000, 0.6000, 0.6000],
                                                [0.7000, 0.7000, 0.8000, 0.8000],
                                                [0.7000, 0.7000, 0.8000, 0.8000]],

                                            [[0.5000, 0.5000, 0.6000, 0.6000],
                                                [0.5000, 0.5000, 0.6000, 0.6000],
                                                [0.7000, 0.7000, 0.8000, 0.8000],
                                                [0.7000, 0.7000, 0.8000, 0.8000]]]]]);
const UPSAMPLE_LINEAR_INPUT     = torch.tensor([[[1, 9, 8, 7]]]);
const UPSAMPLE_LINEAR_RESULT    = torch.tensor([[[1.0, 3.0, 7.0, 8.75, 8.25, 7.75, 7.25, 7.0]]]);
const UPSAMPLE_LINEAR_INPUT7    = torch.tensor([[[0.1, 0.9,  0.8, 0.7]]]);
const UPSAMPLE_LINEAR_RESULT7   = torch.tensor([[[0.1000, 0.1000, 0.1000, 0.1000, 0.2143, 0.3286, 0.4429, 0.5571,
                                                    0.6714, 0.7857, 0.9000, 0.8857, 0.8714, 0.8571, 0.8429, 0.8286,
                                                    0.8143, 0.8000, 0.7857, 0.7714, 0.7571, 0.7429, 0.7286, 0.7143,
                                                    0.7000, 0.7000, 0.7000, 0.7000]]]);


async function test_upsample() {
    let input = UPSAMPLE_INPUT;
    let input_data = await input.toArrayAsync();
    console.log("🔨 Testing upsample: nearest neighbor with input: ", input_data);
    let up = new torch.nn.UpSample(null, 2, "nearest");
    let output = up.forward(input);
    await check(output, UPSAMPLE_RESULT);

    input = UPSAMPLE_2D_INPUT;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing upsample: nearest neighbor with input: ", input_data);
    up = new torch.nn.UpSample(null, [2, 3], "nearest");
    output = up.forward(input);
    await check(output, UPSAMPLE_2D_RESULT);

    input = UPSAMPLE_3D_INPUT;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing upsample: nearest neighbor with input: ", input_data);
    up = new torch.nn.UpSample(null, 2, "nearest");
    output = up.forward(input);
    await check(output, UPSAMPLE_3D_RESULT);

    input = UPSAMPLE_LINEAR_INPUT;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing upsample: linear with input: ", input_data);
    up = new torch.nn.UpSample(null, 2, "linear");
    output = up.forward(input);
    await check(output, UPSAMPLE_LINEAR_RESULT);

    input = UPSAMPLE_LINEAR_INPUT7;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing upsample: linear with input: ", input_data);
    up = new torch.nn.UpSample(null, 7, "linear");
    output = up.forward(input);
    await check(output, UPSAMPLE_LINEAR_RESULT7);
}

const MAXPOOL2D_INPUT       = torch.tensor([[[[ 0.2399,  0.6513,  1.1319,  0.9944],
                                            [ 0.8470,  1.5732,  1.4865,  0.0642],
                                                [-0.7593,  1.8717, -0.4587, -0.9404],
                                                [-1.4540,  0.6457,  0.3717,  0.2037],
                                                [ 1.0059, -0.8514,  1.0126,  0.5497],
                                                [ 0.2664,  2.1315, -2.1271,  1.3594],
                                                [-1.2629, -0.1549, -0.0597, -0.1592],
                                                [-0.2809, -0.2284, -0.4601, -1.9731],
                                                [ 0.6517,  0.5828, -1.2065, -0.1256]],

                                            [[-0.3146,  1.5897, -0.4114,  0.3162],
                                                [ 0.5285, -0.8819,  1.0266,  0.0691],
                                                [-1.1977,  0.4838,  0.8002,  1.9655],
                                                [ 1.0166,  1.0853,  0.9439, -0.4477],
                                                [ 0.0466, -0.3367,  0.5018, -1.1010],
                                                [ 0.0835,  0.2410, -0.3568,  0.1079],
                                                [-1.1327, -0.8702,  0.1342,  0.5102],
                                                [-0.7014,  0.0166,  0.8815,  2.0298],
                                                [-0.6317, -0.0110, -1.7119, -1.4818]]],


                                            [[[ 0.3789, -1.2963, -0.7755, -0.7084],
                                                [-1.4870, -0.5395, -0.7195,  0.2974],
                                                [-0.0076, -0.8185,  0.4957, -0.7257],
                                                [ 0.5529, -0.0276,  0.2878,  0.2377],
                                                [-1.3827, -0.3854, -0.2647, -0.7057],
                                                [-0.6740, -0.4034,  1.3084, -0.9530],
                                                [-0.0480,  1.2390, -0.1759,  1.1122],
                                                [-1.2252, -0.0111,  1.9883,  0.5453],
                                                [ 1.8691,  1.0666,  0.4052,  0.9302]],

                                            [[-0.9603, -0.6785, -1.4000,  0.8593],
                                                [-0.0716,  0.6242, -0.2594, -0.4239],
                                                [-1.7952, -1.0594, -1.6827,  1.0304],
                                                [-0.2368,  0.7444, -0.6710, -0.6236],
                                                [ 0.8887,  1.3046, -0.4016,  0.0944],
                                                [-0.3473,  1.3773,  0.1407,  0.9476],
                                                [-2.4920, -1.3289,  0.3926, -0.2498],
                                                [ 0.6252, -0.0512,  0.4105, -0.4643],
                                                [ 0.1633,  0.9880, -0.1966,  1.3771]]]]);
const MAXPOOL2D_ACTUAL      = torch.tensor([[[[ 1.8717,  1.4865],
                                                [ 2.1315,  1.3594],
                                                [ 0.6517, -0.0597]],

                                            [[ 1.5897,  1.9655],
                                                [ 1.0853,  0.9439],
                                                [ 0.0166,  2.0298]]],


                                            [[[ 0.3789,  0.4957],
                                                [ 0.5529,  1.3084],
                                                [ 1.8691,  1.9883]],

                                            [[ 0.6242,  1.0304],
                                                [ 1.3773,  0.9476],
                                                [ 0.9880,  1.3771]]]]);
const MAXPOOL2D_STRIDE_ACTUAL   = torch.tensor([[[[ 1.8717,  1.8717,  1.4865],
                                                [ 1.8717,  1.8717,  1.0126],
                                                [ 2.1315,  2.1315,  1.3594],
                                                [ 0.6517,  0.5828, -0.0597]],

                                            [[ 1.5897,  1.5897,  1.9655],
                                                [ 1.0853,  1.0853,  1.9655],
                                                [ 0.2410,  0.5018,  0.5102],
                                                [ 0.0166,  0.8815,  2.0298]]],


                                            [[[ 0.3789,  0.4957,  0.4957],
                                                [ 0.5529,  0.4957,  0.4957],
                                                [ 1.2390,  1.3084,  1.3084],
                                                [ 1.8691,  1.9883,  1.9883]],

                                            [[ 0.6242,  0.6242,  1.0304],
                                                [ 1.3046,  1.3046,  1.0304],
                                                [ 1.3773,  1.3773,  0.9476],
                                                [ 0.9880,  0.9880,  1.3771]]]])
const MAXPOOL2D_DILATION_ACTUAL = torch.tensor([[[[ 1.1319,  1.8717],
                                                [ 1.0126,  1.8717],
                                                [ 1.0126,  0.5828]],

                                            [[ 0.8002,  1.9655],
                                                [ 0.8002,  1.9655],
                                                [ 0.5018,  0.5102]]],


                                            [[[ 0.4957, -0.3854],
                                                [ 0.4957,  1.2390],
                                                [ 1.8691,  1.2390]],

                                            [[ 0.8887,  1.3046],
                                                [ 0.8887,  1.3046],
                                                [ 0.8887,  1.3771]]]]);
const MAXPOOL2D_PADDED_ACTUAL = torch.tensor([[[[ 0.8470,  1.5732,  1.5732,  1.4865,  0.9944],
                                                [ 0.8470,  1.8717,  1.8717,  1.4865,  0.2037],
                                                [ 1.0059,  2.1315,  2.1315,  1.3594,  1.3594],
                                                [ 0.2664,  2.1315,  2.1315,  1.3594,  1.3594],
                                                [ 0.6517,  0.6517,  0.5828, -0.1256, -0.1256]],

                                            [[ 0.5285,  1.5897,  1.5897,  1.0266,  0.3162],
                                                [ 1.0166,  1.0853,  1.0853,  1.9655,  1.9655],
                                                [ 1.0166,  1.0853,  1.0853,  0.9439,  0.1079],
                                                [ 0.0835,  0.2410,  0.8815,  2.0298,  2.0298],
                                                [-0.6317,  0.0166,  0.8815,  2.0298,  2.0298]]],


                                            [[[ 0.3789,  0.3789, -0.5395,  0.2974,  0.2974],
                                                [ 0.5529,  0.5529,  0.4957,  0.4957,  0.2974],
                                                [ 0.5529,  0.5529,  1.3084,  1.3084,  0.2377],
                                                [-0.0480,  1.2390,  1.9883,  1.9883,  1.1122],
                                                [ 1.8691,  1.8691,  1.9883,  1.9883,  0.9302]],

                                            [[-0.0716,  0.6242,  0.6242,  0.8593,  0.8593],
                                                [-0.0716,  0.7444,  0.7444,  1.0304,  1.0304],
                                                [ 0.8887,  1.3773,  1.3773,  0.9476,  0.9476],
                                                [ 0.6252,  1.3773,  1.3773,  0.9476,  0.9476],
                                                [ 0.6252,  0.9880,  0.9880,  1.3771,  1.3771]]]]);

async function test_maxpool2d() {
    let input = MAXPOOL2D_INPUT;
    let input_data = await input.toArrayAsync();
    console.log("🔨 Testing maxpool2d with input: ", input_data);
    let mp = new torch.nn.MaxPool2d([3, 2]);
    let output = mp.forward(input);
    await check(output, MAXPOOL2D_ACTUAL);

    console.log("🔨 Testing maxpool2d with stride = [2, 1] & input: ", input_data);
    mp = new torch.nn.MaxPool2d([3, 2], [2, 1]);
    output = mp.forward(input);
    await check(output, MAXPOOL2D_STRIDE_ACTUAL);

    console.log("🔨 Testing maxpool2d with stride = [2, 1], dilation = [2, 2] & input: ", input_data);
    mp = new torch.nn.MaxPool2d([3, 2], [2, 1], [0, 0], [2, 2]);
    output = mp.forward(input);
    await check(output, MAXPOOL2D_DILATION_ACTUAL);

    console.log("🔨 Testing maxpool2d with stride = [2, 1], padding = [1, 1] & input: ", input_data);
    mp = new torch.nn.MaxPool2d([3, 2], [2, 1], [1, 1]);
    output = mp.forward(input);
    await check(output, MAXPOOL2D_PADDED_ACTUAL);
}

const LAYERNORM_INPUT           = torch.tensor([.25, .75]);
const LAYERNORM_ACTUAL          = torch.tensor([-0.9999, 0.9999]);
const LAYERNORM_1D_INPUT        = torch.tensor([0.7646,  0.0049,  1.3850,  0.3942, -1.9543, -0.3378,  0.1121, -0.4909, 0.7837, -0.0792]);
const LAYERNORM_1D_ACTUAL       = torch.tensor([ 0.8194, -0.0619,  1.5390,  0.3897, -2.3345, -0.4594,  0.0625, -0.6370, 0.8415, -0.1594]);
const LAYERNORM_2D_INPUT        = torch.tensor([[-1.0849, -0.1526,  2.7418],
                                                [-1.0062, -0.0444, -1.7595],
                                                [ 0.2015, -0.6739, -1.9398],
                                                [-1.3699,  0.0935,  0.4324],
                                                [ 0.2189, -0.3939, -1.0455],
                                                [-0.4335, -1.3622, -1.6785],
                                                [-0.8717,  0.0477, -0.2336],
                                                [-1.4260,  1.0530,  1.4248],
                                                [-0.2396, -0.7806, -1.1494],
                                                [ 1.0519, -0.9960, -0.0389]]);
const LAYERNORM_2D_NORM1_ACTUAL = torch.tensor([[-0.9737, -0.4014,  1.3751],
                                                [-0.0990,  1.2712, -1.1722],
                                                [ 1.1440,  0.1481, -1.2920],
                                                [-1.3919,  0.4793,  0.9126],
                                                [ 1.2120,  0.0251, -1.2371],
                                                [ 1.3713, -0.3863, -0.9850],
                                                [-1.3497,  1.0405,  0.3092],
                                                [-1.4040,  0.5551,  0.8489],
                                                [ 1.2943, -0.1536, -1.1406],
                                                [ 1.2505, -1.1972, -0.0533]]);
const LAYERNORM_2D_NORM2_ACTUAL = torch.tensor([[-0.6924,  0.2240,  3.0691],
                                                [-0.6150,  0.3304, -1.3555],
                                                [ 0.5721, -0.2884, -1.5327],
                                                [-0.9725,  0.4659,  0.7991],
                                                [ 0.5892, -0.0132, -0.6537],
                                                [-0.0521, -0.9650, -1.2759],
                                                [-0.4828,  0.4209,  0.1444],
                                                [-1.0277,  1.4091,  1.7746],
                                                [ 0.1385, -0.3933, -0.7558],
                                                [ 1.4080, -0.6050,  0.3358]]);

export async function test_layernorm() {
    let input = LAYERNORM_INPUT;
    let input_data = await input.toArrayAsync();
    console.log("🔨 Testing layer norm with input: ", input_data);
    let output = torch.layernorm(input, [2]);
    await check(output, LAYERNORM_ACTUAL);

    input = LAYERNORM_1D_INPUT;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing layer norm 1d with input: ", input_data);
    output = torch.layernorm(input, [10], 0);
    await check(output, LAYERNORM_1D_ACTUAL);
    
    input = LAYERNORM_2D_INPUT;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing layer norm 2d, norm 1 with input: ", input_data);
    output = torch.layernorm(input, [3]);
    await check(output, LAYERNORM_2D_NORM1_ACTUAL);

    input = LAYERNORM_2D_INPUT;
    input_data = await input.toArrayAsync();
    console.log("🔨 Testing layer norm 2d, norm 2 with input: ", input_data);
    output = torch.layernorm(input, [10, 3]);
    await check(output, LAYERNORM_2D_NORM2_ACTUAL);
}

export async function test_conv() {
    const a = randn([10, 10, 10, 4]);
    const a_data = await a.toArrayAsync();
    const b = randn([10, 10, 10, 4]);
    const b_data = await a.toArrayAsync();
    console.log("🔨 Testing repeat with input: ", a, " and : ", b);
    const output = torch.conv2d(a, b);
    const output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);
}

export async function test_repeat() {
    const input = randn([3]);
    const input_data = await input.toArrayAsync();
    const shape = [4, 2];
    console.log(`🔨 Testing repeat with shape, ${shape} and input: `, input);
    const output = torch.repeat(input, shape);
    const output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);
}

export async function test_cat() {
    const a = randn([3, 3]);
    const a_data = await a.toArrayAsync();
    const b = randn([3, 3]);
    const b_data = await b.toArrayAsync();
    console.log(`🔨 Testing cat with input: `, a_data, b_data);
    const output = torch.cat(a, b, 0);
    const output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);
}

export async function test_minmax() {
    let input = randn([10]);
    input = torch.scalar_mul(input, 10);
    const input_data = await input.toArrayAsync();
    console.log(`🔨 Testing min with input: `, input_data);
    let output = torch.min(input, 4);
    let output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);

    console.log(`🔨 Testing max with input: `, input_data);
    output = torch.max(input, 6);
    output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);
}

export async function test_scalars() {
    const inputa = randn([10]);
    const input_data = await inputa.toArrayAsync();
    console.log(`🔨 Testing scalar multiplication with input: `, input_data);
    let output = torch.scalar_mul(inputa, 3);
    let output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);

    console.log(`🔨 Testing scalar division with input: `, input_data);
    output = torch.scalar_div(inputa, 3);
    output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);

    console.log(`🔨 Testing scalar addition with input: `, input_data);
    output = torch.scalar_add(inputa, 5);
    output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);

    console.log(`🔨 Testing scalar subtraction with input: `, input_data);
    output = torch.scalar_sub(inputa, 2);
    output_data = await output.toArrayAsync();
    console.log("=> result: ", output_data);
}

export async function test_linspace() {
    const start = Math.random();
    const end = start + (Math.random() * (1.0 - start));
    const steps = Math.floor(Math.random() * 100);
    console.log(`🔨 Testing linspace with start: ${start}, end: ${end}, and steps: ${steps}`);
    const res = linspace(start, end, steps);
    const data = await res.toArrayAsync();
    console.log("Result: \n", data, "\n---\n");
}
export async function test_randint() {
    const low = Math.floor(Math.random() * 10);
    const high = low + Math.floor(Math.random() * (20 - low));
    const size = 10;
    console.log(`🔨 Testing randint with low: ${low}, high: ${high}, and size: ${size}`);
    const res = randint(low, high, size);
    const data = await res.toArrayAsync();
    console.log("Result: \n", data, "\n---\n");
}
export async function test_randn() {
    const shape: torch.Shape = [
        Math.floor(Math.random()*9 + 1), 
        Math.floor(Math.random()*9 + 1), 
        Math.floor(Math.random()*9 + 1),
        Math.floor(Math.random()*9 + 1)
    ];
    console.log(`🔨 Testing randn with shape: `, shape);
    const res = randn(shape);
    const data = await res.toArrayAsync();
    console.log("Result: \n", data, "\n---\n");
}
export async function test_cumprod() {
    const input = randint(1, 10, 10);
    const input_data = await input.toArrayAsync();
    console.log(`🔨 Testing cumprod with input: \n`, input_data);
    const res = await cumprod(input);
    const data = await res.toArrayAsync();
    console.log("Result: \n", data, "\n---\n");
}

export async function test_clamp() {
    const input = torch.mul(randn([10]), torch.ones([10]), 10);
    const input_data = await input.toArrayAsync();
    console.log(`🔨 Testing clamp on range [-1, 1] and input: \n`, input_data);
    const res = await clamp(input, -1, 1);
    const data = await res.toArrayAsync();
    console.log("Result: \n", data, "\n---\n");
}