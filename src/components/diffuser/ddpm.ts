
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

    constructor(options = {
        noise_steps: 1000,
        beta_start: 0.0001,
        beta_end: 0.02,
        img_size: 64
    }) {
        this.noise_steps = options.noise_steps;
        this.beta_start = options.beta_start;
        this.beta_end = options.beta_end;
        this.img_size = options.img_size;
        
        this.beta = this.prepare_noise_schedule();
        this.alpha = torch.sub(torch.ones(this.noise_steps), this.beta);
        this.alpha_hat = torch.zeros(this.noise_steps);
        this.calc_alpha_hat();
    }

    prepare_noise_schedule() {
        return torch.linspace(this.beta_start, this.beta_end, this.noise_steps);
    }

    async calc_alpha_hat() {
        this.alpha_hat = await torch.cumprod(this.alpha);
    }

    sample_timesteps(n: number) {
        return torch.randint(1, this.noise_steps, [n]);
    }

    sample(model, n=1): torch.Tensor {
        console.log(`Sampling ${n} new images...`);
        model.eval();
        let x = torch.randn([n, 3, this.img_size, this.img_size]);
        for(let i = this.noise_steps; i >= 0; i--) {
            const t = torch.mul(torch.ones(n), torch.ones(n), i);
            const predicted_noise = model.forward(x, t);
            const alpha = this.alpha;
            const alpha_hat = this.alpha_hat
            const beta = this.beta
            let noise;
            if(i > 1) {
                noise = torch.randn(x.shape);
            } else {
                noise = torch.zeros(x.shape);
            }
            x = torch.add(
                    torch.mul(
                        torch.div(torch.ones(alpha.shape), torch.sqrt(alpha)),
                        torch.sub(x, 
                            torch.mul(
                                torch.div(
                                    torch.sub(torch.ones(alpha.shape), alpha),
                                    torch.sqrt(torch.sub(torch.ones(alpha_hat.shape), alpha_hat))
                                ),
                                predicted_noise
                            )
                        )
                    ),
                    torch.mul(
                        torch.sqrt(beta),
                        noise
                    )
                );
        }
        
        model.train();
        x = torch.scalar_div(torch.scalar_add(torch.clamp(x, -1, 1), 1), 2);
        x = torch.scalar_mul(x, 255)
        return x;
    }

}