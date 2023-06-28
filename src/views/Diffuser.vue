<template>
    <div>
        <h2>Diffuser</h2>
        <button @click="generate">go !!</button> <br />
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
    mounted: async function() {
        await init_device();
    },
    methods: {
        generate: async function() {
            const diffuser = new Diffusion({ noise_steps: 1, img_size: 64 });
            const model = new UNet();
            let res = diffuser.sample(model);
            res = res.cat(torch.constant([1, 1, ...Array.from(res.shape).splice(2)], 255), 1);
            res = res.transpose(1, 2).transpose(2, 3);
            console.log("res shape: ", res.shape);
            const res_data = await res.toArrayAsync();
            console.log("sample result: ");
            console.log(res_data);
            const img_data = new Uint8ClampedArray(res_data.flat(3) as any);
            const context = this.$refs["canvas"].getContext("2d");
            context.putImageData(
                new ImageData(img_data, 64, 64), 
                0, 0);
        }
    }
})
</script>