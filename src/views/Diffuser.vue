<template>
    <div>
        <h2>Web Diffusion</h2>
        <div v-if="device_available">
            Noise Steps: <input type="number" v-model="noiseSteps" /> <br>
            Select Weights: <button @click="loadPokemon" :disabled="active">Pokemon</button> <span v-if="done_loading_weights">✅</span><br>
            <button @click="gen">generate</button><br><br>

            <div ref="cycle-list" class="cycle-list">

            </div>
        </div>
        <div v-else>
            Sorry, WebDiffsuion requires WebGPU to be available :(
        </div>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue"
import { init_device } from "@/components/device";
import { UNet } from "@/components/diffuser/modules"
import { Diffusion } from "@/components/diffuser/ddpm"
import * as torch from "@/torch";

let model = null

export default defineComponent({
    name: "Diffuser",
    data() {
        return {
            device_available: true,
            active: false,
            done_loading_weights: false,
            noiseSteps: 1000,
        }
    },
    mounted: async function() {
        this.device_available = await init_device();
        this.model = new UNet();
    },
    methods: {
        loadPokemon: async function() {
            if(!this.active) {
                console.log("loading weights...");
                this.active = true;
                model = new UNet();
                await model.loadStateDictFromURL("https://web-diffusion-worker.0xtimmy.workers.dev/parameters/pokemon");
                console.log("✅ done");
                this.done_loading_weights = true;
                this.active = false;
            }
        },
        gen: async function() {
            if(!this.active) {
                this.active = true;
                const diffuser = new Diffusion({ noise_steps: this.noiseSteps, img_size: 64 });
                const res = await diffuser.sample(model, async (res: torch.Tensor, step_num: number) => { 
                    await this.renderResult(res, `Iteration ${step_num}`);
                    return;
                });
                this.renderResult(res, "final");
                this.active = false;
            }
        },
        renderResult: async function(result: torch.Tensor, caption: string) {
            result = result.squeeze(0).transpose(0, 1).transpose(1, 2);
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
            this.$refs["cycle-list"].replaceChildren(box);
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