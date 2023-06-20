<template>
    <div>
        <h2>Tests</h2>
        <input type="file" value="" @change="runTestfile">
        <p>open the console -></p>
    </div>
</template>

<script lang="ts">
import { defineComponent } from "vue"
import { init_device } from "@/components/device";
import tester from "@/scripts/tester";

export default defineComponent({
    name: "Diffuser",
    mounted: async function() {
        await init_device();
    },
    methods: {
        runTestfile(event) {
            console.log(event);
            const reader = new FileReader();
            reader.onload = (event) => {
                const json = JSON.parse(event.target.result as string);
                tester(json);
            }
            reader.readAsText(event.target.files[0])
        }
    }
})
</script>