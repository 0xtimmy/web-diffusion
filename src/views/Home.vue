<template>
  <div class="home">
    <h1>Web Diffuser</h1>
    <aside class="notes">
      <h2>Notes</h2>
    </aside>
    <main>
      <h2>Testing Ground</h2>
      <Diffuser />
    </main>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import Diffuser from "@/components/Diffuser.vue"

export default defineComponent({
  name: "Home",
  components: {
    Diffuser
  }
});
</script>

<style>

.notes {
  float: right;
  background-color: #ffffff;
}

</style>
