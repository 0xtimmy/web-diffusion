<template>
  <div class="navbar">
    <router-link to="/" class="nav-item">Diffuser</router-link>
    <router-link to="/performance" class="nav-item">Performance</router-link>
    <router-link to="/tests" class="nav-item">Tests</router-link>
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
