<template>
    <div>
        <h2>Diffuser</h2>
        <button @click="generate">go !!</button>
        <canvas ref="canvas">

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
        generate: function() {
            const diffuser = new Diffusion({ noise_steps: 1, img_size: 64 });
            const model = new UNet();
            let res = diffuser.sample(model);
            res = res.cat(torch.ones([1, 1, ...Array.from(res.shape).splice(2)]), 1);
            console.log("res shape: ", res.shape);
            res = res.transpose(1, 2).transpose(2, 3);
            console.log("res shape: ", res.shape);
            (async () => {

                const res_data = await res.toArrayAsync();
                console.log("sample result: ");
                console.log(res_data);
                const img_data = new Uint8ClampedArray(res_data.flat(3) as any);
                /*
                const arr = new Uint8ClampedArray(4 * 200 * 200);
                for (let i = 0; i < arr.length; i += 4) {
                    arr[i + 0] = 0; // R value
                    arr[i + 1] = 0; // G value
                    arr[i + 2] = 0; // B value
                    arr[i + 3] = 255; // A value
                }
                */
               console.log(img_data);
                const img = new ImageData(img_data, res.shape[1], res.shape[2]);
                console.log(img);
                const context = this.$refs["canvas"].getContext("2d");
                context.putImageData(img, 16, 16);
            })()
        }
    }
})
</script>