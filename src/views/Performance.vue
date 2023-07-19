<template>
    <div>
        <h2>Performance Measurements</h2>
        Noise steps: <input type="number" v-model="num_steps" /> <br>
        <button @click="dryrun">dry run</button>
    </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import * as torch from "@/torch"
import { init_device } from "@/components/device";
import { UNet } from '@/components/diffuser/modules';
import { Diffusion } from '@/components/diffuser/ddpm';
import { report_durations } from '@/torch';
import { report_kernel_stats } from '@/torch/kernel_webgpu';

export default defineComponent({
    name: "Performance",
    data() {
        return {
            num_steps: 10,
        }
    },
    mounted: async function() {
        await init_device();
    },
    methods: {
        dryrun: async function() {
            const model = new UNet()
            const diffuser = new Diffusion({ noise_steps: this.num_steps, img_size: 64 });
            let lastIteration = Date.now();
            const res = await diffuser.sample(model, async (res: torch.Tensor, step_num: number) => { 
                const finished_at = Date.now();
                console.log(`⏰ iteration took ${finished_at-lastIteration}ms`);
                lastIteration = finished_at;
                return;
            });
            report_durations();
            report_kernel_stats();
        }
    }
})
</script>