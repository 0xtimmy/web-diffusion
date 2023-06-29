<template>
    <div>
        <h2>Diffuser</h2>
        <button @click="generate" :disabled="active">go !!</button> <br />

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
            active: false,
            
        }
    },
    mounted: async function() {
        await init_device();
    },
    methods: {
        generate: function() {
            if(!this.active) {
                this.active = true;
                const diffuser = new Diffusion({ noise_steps: 1, img_size: 64 });
                const model = new UNet();
                const res = diffuser.sample(model, this.renderResult);
                this.active = false;
                this.renderResult(res);
            }
        },
        renderResult: function(result: torch.Tensor) {
            result = result.cat(torch.constant([1, 1, ...Array.from(result.shape).splice(2)], 255), 1);
            result = result.transpose(1, 2).transpose(2, 3);
            result.toArrayAsync().then((res_data) => {
                if(!this.active) console.log("Result: ", res_data);
                const img_data = new Uint8ClampedArray(res_data.flat(3) as any);
                const context = this.$refs["canvas"].getContext("2d");
                context.putImageData(
                    new ImageData(img_data, 64, 64), 
                    0, 0);
            })
        }
    }
})
</script>