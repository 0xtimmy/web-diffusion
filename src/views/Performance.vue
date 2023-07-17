<template>
    <div>
        <h2>Performance Measurements</h2>
        <button @click="dryrun">run</button>
    </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import * as torch from "@/torch"
import { init_device } from "@/components/device";
import { UNet } from '@/components/diffuser/modules';
import { Diffusion } from '@/components/diffuser/ddpm';

export default defineComponent({
    name: "Performance",
    mounted: async function() {
        await init_device();
    },
    methods: {
        dryrun: async function() {
            const model = new UNet()
            const diffuser = new Diffusion({ noise_steps: 10, img_size: 64 });
            let lastIteration = Date.now();
            const res = await diffuser.sample(model, async (res: torch.Tensor, step_num: number) => { 
                const finished_at = Date.now();
                console.log(`⏰ iteration took ${finished_at-lastIteration}ms`);
                lastIteration = finished_at;
                return;
            });
        }
    }
})
</script>