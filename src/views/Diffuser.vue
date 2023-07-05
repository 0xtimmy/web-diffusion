<template>
    <div>
        <h2>Diffuser</h2>
        Select Weights: <button @click="loadPokemon" :disabled="weightsSelected">Pokemon</button>
        <button @click="generate" :disabled="active || !modelReady">go !!</button> <br />

        <canvas ref="canvas" width="64" height="64">

        </canvas>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue"
import { init_device } from "@/components/device";
import { Diffusion } from "@/components/diffuser/ddpm"
import { UNet } from "@/components/diffuser/modules"
import * as torch from "@/torch";

export default defineComponent({
    name: "Diffuser",
    data() {
        return {
            modelReady: true,
            active: false,
            weightsSelected: false,
            model: null,
        }
    },
    mounted: async function() {
        await init_device();
        this.model = new UNet();
    },
    methods: {
        loadPokemon: async function(event) {
            this.weightsSelected = true;
            const reader = new FileReader();
            console.log("loading weights...");
            await this.model.loadStateDictFromURL("../../parameters/pokemon");
            console.log("✅ done loading weights");
            this.modelReady = true;
            //this.generate();
        },
        generate: async function() {
            if(!this.active) {
                this.active = true;
                const diffuser = new Diffusion({ noise_steps: 100, img_size: 64 });
                const res = await diffuser.sample(this.model, async (res: torch.Tensor) => { 
                    await this.renderResult(res)
                });
                this.active = false;
                this.renderResult(res);
            }
        },
        renderResult: async function(result: torch.Tensor) {
            result = result.cat(torch.constant([1, 1, ...Array.from(result.shape).splice(2)], 255), 1);
            result = result.transpose(1, 2).transpose(2, 3);
            const data = await result.toArrayAsync()
            if(!this.active) console.log("Result: ", data);
            const img_data = new Uint8ClampedArray(data.flat(4) as any);
            const context = this.$refs["canvas"].getContext("2d");
            context.putImageData(
                new ImageData(img_data, 64, 64), 
                0, 0);
        }
    }
})
</script>