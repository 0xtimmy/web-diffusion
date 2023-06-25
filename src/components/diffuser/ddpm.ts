
import * as torch from "../../torch"

export 

class Diffusion {
    noise_steps: number;
    beta_start: number;
    beta_end: number;
    img_size: number;

    beta: torch.Tensor;
    alpha: torch.Tensor;
    alpha_hat: torch.Tensor;

    constructor({
        noise_steps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        img_size=64
    }) {
        this.noise_steps = noise_steps;
        this.beta_start = beta_start;
        this.beta_end = beta_end;
        this.img_size = img_size;
        
        this.beta = this.prepare_noise_schedule();
        this.alpha = torch.sub(torch.ones(this.noise_steps), this.beta);
        this.alpha_hat = torch.zeros(this.noise_steps);
        this.calc_alpha_hat();
    }

    prepare_noise_schedule() {
        return torch.linspace(this.beta_start, this.beta_end, this.noise_steps);
    }

    calc_alpha_hat() {
        this.alpha_hat = torch.cumprod(this.alpha);
    }

    sample_timesteps(n: number) {
        return torch.randint(1, this.noise_steps, [n]);
    }

    sample(model, n=1): torch.Tensor {
        console.log(`Sampling ${n} new images...`);
        model.eval();
        let x = torch.normal([n, 3, this.img_size, this.img_size]);
        for(let i = this.noise_steps -1; i >= 0; i--) {
            const t = torch.scalar_mul(torch.ones(n), i);
            let predicted_noise = model.forward(x, t);
            
            //let predicted_noise = torch.normal([1, 3, 64, 64]);
            // makes it out of the unet
            (async () => { console.log("predicted noise init: ", await predicted_noise.toArrayAsync()) })();
            const alpha = this.alpha.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            const alpha_hat = this.alpha_hat.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            const beta = this.beta.index(t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
            let noise;
            if(i > 1) {
                noise = torch.randn(x.shape);
            } else {
                noise = torch.zeros(x.shape);
            }
            console.log("shapes: x, alpha, alpha_hat, beta", x.shape, alpha.shape, alpha_hat.shape, beta.shape);

            let one_div_sqrt_alpha = torch.div(torch.ones(alpha.shape), torch.sqrt(alpha));
            (async () => { console.log("one_div_sqrt_alpha: ", await one_div_sqrt_alpha.toArrayAsync()) })();
            let sqrt_one_minus_alpha_hat = torch.sqrt(torch.scalar_add(torch.scalar_mul(alpha_hat, -1), 1));
            (async () => { console.log("sqrt_one_minus_alpha_hat: ", await sqrt_one_minus_alpha_hat.toArrayAsync()) })();
            let one_minus_alpha = torch.scalar_add(torch.scalar_mul(alpha_hat, -1), 1);
            (async () => { console.log("one_minus_alpha: ", await one_minus_alpha.toArrayAsync()) })();
            predicted_noise = torch.mul(predicted_noise, torch.div(one_minus_alpha, sqrt_one_minus_alpha_hat));
            (async () => { console.log("predicted_noise: ", await predicted_noise.toArrayAsync()) })();
            x = torch.sub(x, predicted_noise);
            (async () => { console.log("x - predicted_noise: ", await x.toArrayAsync()) })();
            x = torch.mul(one_div_sqrt_alpha, x);
            (async () => { console.log("x * one_div_sqrt_alpha: ", await x.toArrayAsync()) })();
            
            let beta_noise = torch.mul(torch.sqrt(beta), noise);
            (async () => { console.log("beta_noise: ", await beta_noise.toArrayAsync()) })();
            x = torch.add(x, beta_noise);
            (async () => { console.log("final x for noise pass : ", 1, await x.toArrayAsync()) })();
        }
        
        model.train();
        x = torch.scalar_div(torch.scalar_add(torch.clamp(x, -1, 1), 1), 2);
        (async () => { console.log("x / 2 : ", 1, await x.toArrayAsync()) })();
        x = torch.scalar_mul(x, 255)
        return x;
    }

}