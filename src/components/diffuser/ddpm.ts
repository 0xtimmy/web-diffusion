
import * as torch from "@/torch"

export class Diffusion {
    noise_steps: number;
    beta_start: number;
    beta_end: number;
    img_size: number;

    beta: torch.Tensor;
    alpha: torch.Tensor;
    alpha_hat: torch.Tensor;

    constructor({
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64
    }) {
        this.noise_steps = noise_steps;
        this.beta_start = beta_start;
        this.beta_end = beta_end;
        this.img_size = img_size;
        
        this.beta = this.prepare_noise_schedule();
        this.alpha = torch.scalar_mul(this.beta, -1).scalar_add(1);
        this.alpha_hat = torch.cumprod(this.alpha);
    }

    prepare_noise_schedule() {
        return torch.linspace(this.beta_start, this.beta_end, this.noise_steps);
    }

    sample_timesteps(n: number) {
        return torch.randint(1, this.noise_steps, [n]);
    }

    async sample(model, handleStep?: (img: torch.Tensor, step_num: number) => void, n=1): Promise<torch.Tensor> {
        console.log(`Sampling ${n} new images...`);
        const sampleStart = Date.now();
        //model.eval();
        const t_x = Date.now();
        let x = torch.randn([n, 3, this.img_size, this.img_size]);
        let total_model_time = 0;
        //console.log(`⏰ initial x generation took ${Date.now()-t_x}ms`);
        for(let i = this.noise_steps-1; i > 0; i--) {
            const t = torch.constant([n], i);
            const t_predicted_noise = Date.now();
            let predicted_noise = model.forward(x, t);
            //console.log(`⏰ model.foward took ${Date.now()-t_predicted_noise}ms`);
            total_model_time += Date.now()-t_predicted_noise;
            
            const t_indexing = Date.now();
            const alpha = torch.index(this.alpha, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            const alpha_hat = torch.index(this.alpha_hat, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            const beta = torch.index(this.beta, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            //console.log(`⏰ indexing, alpha, alpha_hat, beta took ${Date.now()-t_indexing}ms`);

            const t_noise = Date.now();
            let noise;
            if(i > 1) {
                noise = torch.randn(x.shape);
            } else {
                noise = torch.zeros(x.shape);
            }      
            //console.log(`⏰ noise gen took ${Date.now()-t_noise}ms`);

            const t_denoising = Date.now();
            let one_div_sqrt_alpha = torch.sqrt(alpha).scalar_pow(-1);
            let sqrt_one_minus_alpha_hat = alpha_hat.scalar_mul(-1).scalar_add(1).sqrt();
            let one_minus_alpha = alpha.scalar_mul(-1).scalar_add(1);
            const alpha_div_alpha_hat = torch.div(one_minus_alpha, sqrt_one_minus_alpha_hat);
            one_minus_alpha.destroy();
            sqrt_one_minus_alpha_hat.destroy();
            predicted_noise = predicted_noise.mul(alpha_div_alpha_hat);
            alpha_div_alpha_hat.destroy();
            let nx = torch.sub(x, predicted_noise);
            
            nx = nx.mul(one_div_sqrt_alpha);
            one_div_sqrt_alpha.destroy();
            
            const beta_noise = beta.sqrt().mul(noise);
            noise.destroy();
            
            nx = nx.add(beta_noise);
            beta_noise.destroy();
            //console.log(`⏰ denoising took ${Date.now()-t_denoising}ms`);
            
            console.log(`${(this.noise_steps-i)/this.noise_steps*100}% - ${Date.now() - sampleStart}ms`);

            //console.log("x and nx diff: ", array_eq((await x.toArrayAsync()).flat(4), (await nx.toArrayAsync()).flat(4)));

            x = nx;
            
            if(typeof(handleStep) != 'undefined') {
                let step = torch.clamp(x, -1, 1).scalar_add(1).scalar_div(2)
                step = step.cat(torch.ones([1, 1, ...Array.from(x.shape).splice(2)]), 1);
                step = step.scalar_mul(255)
                await handleStep(step, this.noise_steps-i);
            }
        }
        //model.train();
        console.log(`🟩 model took on average ${total_model_time / (this.noise_steps-1)}ms`);
        return x.clamp(-1, 1).scalar_add(1).scalar_div(2).cat(torch.ones([1, 1, ...Array.from(x.shape).splice(2)]), 1).scalar_mul(255);
    }

    async _sample(model, x: torch.Tensor, noises: Array<any>, results: Array<any>, handleStep: (img: torch.Tensor) => void, n=1): Promise<torch.Tensor> {
        console.log(`Sampling ${n} new images...`);
        const sampleStart = Date.now();
        //model.eval();
        //x = torch.randn([n, 3, this.img_size, this.img_size]).scalar_add(0.2);
        for(let i = this.noise_steps-1; i > 0; i--) {
            const t = torch.constant([n], i);
            let predicted_noise = model.forward(x, t);
            //let predicted_noise = x.copy();
            
            const alpha = torch.index(this.alpha, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            const alpha_hat = torch.index(this.alpha_hat, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            const beta = torch.index(this.beta, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);

            let noise;
            //if(i > 1) {
            //    noise = torch.randn(x.shape);
            //} else {
            //    noise = torch.zeros(x.shape);
            //}          
            //await comp_stats(noise, torch.tensor(noises[i]));
            noise = torch.tensor(noises[i]);    
            let one_div_sqrt_alpha = torch.sqrt(alpha).scalar_pow(-1);
            
            let sqrt_one_minus_alpha_hat = alpha_hat.scalar_mul(-1).scalar_add(1).sqrt();
            let one_minus_alpha = alpha.scalar_mul(-1).scalar_add(1);
            const alpha_div_alpha_hat = torch.div(one_minus_alpha, sqrt_one_minus_alpha_hat);
            one_minus_alpha.destroy();
            sqrt_one_minus_alpha_hat.destroy();
            predicted_noise = predicted_noise.mul(alpha_div_alpha_hat);
            alpha_div_alpha_hat.destroy();
            let nx = torch.sub(x, predicted_noise);
            
            nx = nx.mul(one_div_sqrt_alpha);
            one_div_sqrt_alpha.destroy();
            
            const beta_noise = beta.sqrt().mul(noise);
            noise.destroy();
            
            nx = nx.add(beta_noise);
            beta_noise.destroy();
            
            console.log(`${(this.noise_steps-i)/this.noise_steps*100}% - ${Date.now() - sampleStart}ms`);

            console.log("x and nx diff: ", array_eq((await x.toArrayAsync()).flat(4), (await nx.toArrayAsync()).flat(4)));

            x = nx;
            
            let result = torch.clamp(x, -1, 1).scalar_add(1).scalar_div(2)
            result = torch.cat(result, torch.ones([1, 1, ...Array.from(x.shape).splice(2)]), 1);
            result = result.scalar_mul(255)
            await handleStep(result.copy());
            result = result.squeeze(0).transpose(0, 1).transpose(1, 2);
            const actual_output = result;
            const output_data = await actual_output.toArrayAsync();
            const target = results[i];
            const target_output = torch.tensor(target);
            if(array_eq(actual_output.shape, target_output.shape) > 0) console.warn(`‼️ - ${i}`, { res: false, output: output_data, msg: `mismatched shapes-- expected ${target_output.shape}, got ${actual_output.shape}` });
            const diff = array_eq(output_data.flat(4), target.flat(4));
            if(diff > 0.00001) console.warn(`‼️ - ${i}`, { res: false, output: output_data, msg: `mismatched tensor content, average diff: ${diff}` });
            else console.log(`✅ - ${i}`);
        }
        //model.train();
        return x.clamp(-1, 1).scalar_add(1).scalar_mul(255/2);
    }
}

function array_eq(a: Array<any>, b: Array<any>): number {
    if(a.length != b.length) return Infinity;
    const diff = (a.reduce((acc, v, i) => {
        return acc + Math.abs(v - b[i]);
    }, 0) / a.length);
    if(isNaN(diff)) return Infinity;
    return diff;
}

async function comp_stats(a: torch.Tensor, b: torch.Tensor) {
    let output_data = await a.toArrayAsync();
    output_data = output_data.flat(4)

    let target = await b.toArrayAsync();
    target = target.flat();

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
    ) console.warn(`mismatched stats, target mean=${target_mean}, target var=${target_variance}, target std=${target_std}; actual mean=${actual_mean}, actual var=${actual_variance}, actual std=${actual_std}`);
    else console.log("✅ noises similair");
}