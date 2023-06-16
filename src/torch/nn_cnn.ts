import { Module } from "./nn_module";
import { Tensor } from "./tensor"

export class AvgPooling2d extends Module {}

export class Conv2d extends Module {
    inChannels: number;
    outChannels: number;
    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | [number, number],
        stride: number | [number, number],
        padding: number | [number, number] | "valid" | "same",
        dtype: string
    ) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
    }

    forward(input: Tensor) {
        console.error("trying to forward conv2d")
    }
}

export class ConvTranspose2d extends Module {}

export class GroupNorm extends Module {
    numGroups: number;
    numChannels: number;
    constructor(numGroups: number, numChannels: number) {
        super();
        this.numGroups = numGroups;
        this.numChannels = numChannels;
    }

    forward(input: Tensor) {
        console.error("trying to forward group norm")
    }
}

export class Linear extends Module {
    inChannels: number;
    outChannels: number;
    constructor(inChannels: number, outChannels: number) {
        super();
        this.inChannels = inChannels;
        this.outChannels = outChannels;
    }

    forward(input: Tensor) {
        console.error("trying to forward linear")
    }
}
