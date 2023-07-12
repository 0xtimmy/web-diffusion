<template>
  <div class="navbar">
    <router-link to="/tests" class="nav-item">Tests</router-link>
    <router-link to="/" class="nav-item">Diffuser</router-link>
  </div>
  <router-view />
</template>

<style>
@import "./assets/styles.css";

.navbar {
  display: flex;
  gap: 8px;
}

.nav-item {
  cursor: pointer;
}

</style>
