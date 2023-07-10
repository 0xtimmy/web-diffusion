<template>
    <div>
        <h2>Web Diffusion</h2>
        Select Weights: <button @click="loadPokemon" :disabled="weightsSelected">Pokemon</button> <br>
        <button @click="generate" :disabled="active || !modelReady">go !!</button> <br />

        <div ref="cycle-list" class="cycle-list">

        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue"
import { init_device } from "@/components/device";
import { UNet } from "@/components/diffuser/modules"
import { Diffusion } from "@/components/diffuser/ddpm"
import * as torch from "@/torch";

export default defineComponent({
    name: "Diffuser",
    data() {
        return {
            modelReady: true,
            active: false,
            weightsSelected: false,
            //model: null,
        }
    },
    mounted: async function() {
        await init_device();
        this.model = new UNet();
    },
    methods: {
        loadPokemon: async function(event) {
            this.weightsSelected = true;
            console.log("loading weights...");
            await this.model.loadStateDictFromURL("../../parameters/pokemon");
            console.log("✅ done loading weights");
            this.modelReady = true;
            this.generate();
        },
        generate: async function() {
            if(!this.active) {
                const model = new UNet();
                this.active = true;
                const diffuser = new Diffusion({ noise_steps: 100, img_size: 64 });
                const res = await diffuser.sample(model, async (res: torch.Tensor, step_num: number) => { 
                    await this.renderResult(res, `Iteration ${step_num}`);
                    return;
                });
                console.log(res);
                this.active = false;
                this.renderResult(res, "final");
            }
        },
        renderResult: async function(result: torch.Tensor, caption: string) {
            result = result.cat(torch.constant([1, 1, ...Array.from(result.shape).splice(2)], 255), 1);
            result = result.transpose(1, 2).transpose(2, 3);
            const data = await result.toArrayAsync();
            if(!this.active) console.log("Result: ", data);
            const img_data = new Uint8ClampedArray(data.flat(4) as any);
            const box = document.createElement("div");
            box.className = "result-box";

            const canvas = document.createElement("canvas");
            canvas.setAttribute("width", "64px");
            canvas.setAttribute("height", "64px");
            const context = canvas.getContext("2d");
            context.putImageData(
                new ImageData(img_data, 64, 64), 
                0, 0);
            box.appendChild(canvas);
            const cap = document.createElement("div");
            cap.innerHTML = caption;
            box.appendChild(cap);
            this.$refs["cycle-list"].appendChild(box);
            return;
        }
    }
})
</script>

<style>
.cycle-list {
    display: flex;
    flex-direction: column-reverse;
    gap: 16px;
}

canvas {
    width: 64px;
    height: 64px;
}

.result-box {
    display: flex;
    flex-direction: column;
    align-items: left;
    font-size: 15px;
}
</style>