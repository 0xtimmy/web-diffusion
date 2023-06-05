<template>
  <div class="home">
    <h1>ts-torch</h1>
    <Tests />
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import Tests from "@/components/Tests.vue"

export default defineComponent({
  name: "Home",
  components: {
    Tests
  }
});
</script>

<style>



</style>
