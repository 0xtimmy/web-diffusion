import { Shape } from "./shape"
import { Tensor } from "./tensor"
import { rand } from "./ops_artisanal";
import { empty } from "./factories";

export class Generator {
    seed: number;
    constructor(seed?: number) {
        this.seed = seed ? seed : (Math.random() * Number.MAX_SAFE_INTEGER);
    }

    sample(shape: Shape): Tensor {
        const sampling = rand(empty(shape), this.seed);
        this.seed = sampling.next_seed;
        return sampling.output;
    }
}