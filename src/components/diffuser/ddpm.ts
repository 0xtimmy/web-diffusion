
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
        this.alpha = torch.sub(torch.ones(this.noise_steps), this.beta);
        this.alpha_hat = torch.cumprod(this.alpha);
    }

    prepare_noise_schedule() {
        return torch.linspace(this.beta_start, this.beta_end, this.noise_steps);
    }

    sample_timesteps(n: number) {
        return torch.randint(1, this.noise_steps, [n]);
    }

    sample(model, handleStep?: (img: torch.Tensor, step_num: number) => void, n=1): torch.Tensor {
        console.log(`Sampling ${n} new images...`);
        const sampleStart = Date.now();
        //model.eval();
        let x = torch.normal([n, 3, this.img_size, this.img_size]);
        for(let i = this.noise_steps -1; i >= 0; i--) {
            console.log(`Starting pass #${this.noise_steps-i}`)
            //try {
                const t = torch.constant([n], i);
                let predicted_noise = model.forward(x, t);
                
                const alpha = this.alpha.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
                const alpha_hat = this.alpha_hat.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
                const beta = this.beta.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
                
                let noise;
                if(i > 1) {
                    noise = torch.randn(x.shape);
                } else {
                    noise = torch.zeros(x.shape);
                }
                
                let one_div_sqrt_alpha = torch.div(torch.ones(alpha.shape), torch.sqrt(alpha));
                
                let sqrt_one_minus_alpha_hat = torch.sqrt(torch.sub(torch.ones(alpha_hat.shape), alpha_hat));
                let one_minus_alpha = torch.sub(torch.ones(alpha.shape), alpha);
                predicted_noise = torch.mul(predicted_noise, torch.div(one_minus_alpha, sqrt_one_minus_alpha_hat));
                let nx = torch.sub(x, predicted_noise);
                nx = torch.mul(one_div_sqrt_alpha, nx);
                
                const beta_noise = torch.mul(torch.sqrt(beta), noise);
                nx = torch.add(nx, beta_noise);
                
                console.log(`${(this.noise_steps-i)/this.noise_steps*100}% - ${Date.now() - sampleStart}ms`);

                /*
                const dev = torch.devices["webgpu"] as any;
                const err = await dev.gpuDevice.popErrorScope();
                console.log("error? ", err);
                */

                if(typeof(handleStep) != 'undefined') {
                    handleStep(torch.scalar_mul(torch.scalar_add(torch.clamp(nx, -1, 1), 1), 255/2), (this.noise_steps-i));
                }
                
                x = nx;
                
                /*
            } catch(e: any) {
                console.log("caught error while sampling, retrying ", e);
                if(e == "DOMException: Device is lost") console.log("we got em");
                await torch.initWebGPUAsync();
                i++;
            }
            */
        }
        //model.train();
        return torch.scalar_mul(torch.scalar_add(torch.clamp(x, -1, 1), 1), 255/2);
    }

}