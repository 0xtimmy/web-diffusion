<template>
    <div>
        <h2>Diffuser</h2>
        <button @click="generate">go !!</button>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue"
import { init_device } from "@/components/device";
import { Diffusion } from "@/components/diffuser/ddpm"
import { UNet } from "@/components/diffuser/modules"

export default defineComponent({
    name: "Diffuser",
    mounted: async function() {
        await init_device();
    },
    methods: {
        generate: function() {
            const diffuser = new Diffusion();
            const model = new UNet();
            const res = diffuser.sample(model);
            (async () => {
                const res_data = await res.toArrayAsync();
                console.log("sample result: ");
                console.log(res_data);
            })()
        }
    }
})
</script>