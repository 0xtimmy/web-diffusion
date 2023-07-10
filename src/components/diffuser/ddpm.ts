
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

    async sample(model, handleStep?: (img: torch.Tensor, step_num: number) => void, n=1): Promise<torch.Tensor> {
        console.log(`Sampling ${n} new images...`);
        const sampleStart = Date.now();
        //model.eval();
        let x = torch.normal([n, 3, this.img_size, this.img_size]);
        for(let i = this.noise_steps -1; i >= 0; i--) {
            console.log(`Starting pass #${this.noise_steps-i}`)
            //try {
                const t = torch.constant([n], i);
                let predicted_noise = await model.forward(x, t);
                //let predicted_noise = x.copy();
                
                const alpha = torch.index(this.alpha, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
                const alpha_hat = torch.index(this.alpha_hat, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
                const beta = torch.index(this.beta, t).repeat([1, x.shape[1], x.shape[2], x.shape[3]]);
                t.destroy();

                let noise;
                if(i > 1) {
                    noise = torch.randn(x.shape);
                } else {
                    noise = torch.zeros(x.shape);
                }
                
                /*
                let one_div_sqrt_alpha = torch.sqrt(alpha).scalar_pow(-1);
                
                let sqrt_one_minus_alpha_hat = alpha_hat.scalar_mul(-1).scalar_add(1).sqrt();
                let one_minus_alpha = alpha.scalar_mul(-1).scalar_add(1);
                const alpha_div_alpha_hat = torch.div(one_minus_alpha, sqrt_one_minus_alpha_hat);
                one_minus_alpha.destroy();
                sqrt_one_minus_alpha_hat.destroy();
                predicted_noise = predicted_noise.mul(alpha_div_alpha_hat);
                alpha_div_alpha_hat.destroy();
                let nx = x.sub(predicted_noise);
                
                nx = nx.mul(one_div_sqrt_alpha);
                one_div_sqrt_alpha.destroy();
                
                const beta_noise = beta.sqrt().mul(noise);
                noise.destroy();
                
                nx = nx.add(beta_noise);
                beta_noise.destroy();
                
                console.log(`${(this.noise_steps-i)/this.noise_steps*100}% - ${Date.now() - sampleStart}ms`);

                if(typeof(handleStep) != 'undefined') {
                    handleStep(torch.scalar_mul(torch.scalar_add(torch.clamp(nx, -1, 1), 1), 255/2), (this.noise_steps-i));
                }

                alpha.destroy();
                alpha_hat.destroy();
                beta.destroy();
                
                x = nx;
                */
        }
        //model.train();
        return torch.scalar_mul(torch.scalar_add(torch.clamp(x, -1, 1), 1), 255/2);
    }

}