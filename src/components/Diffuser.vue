<template>
    <div>
        <h3>Diffuser</h3>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue"
import { init_device } from "@/scripts/device";
import tests from "@/scripts/tests";

export default defineComponent({
    name: "Diffuser",
    mounted: async function() {
        await init_device();

        // Tests
        tests();
    },
})
</script>