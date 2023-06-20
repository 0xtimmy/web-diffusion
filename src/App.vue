<template>
  <div class="navbar">
    <router-link to="/" class="nav-item">Tests</router-link>
    <router-link to="/diffuser" class="nav-item">Diffuser</router-link>
  </div>
  <router-view />
</template>

<style>
@import "./assets/styles.css";

.navbar {
  display: flex;
}

.nav-item {
  padding: 5px 8px;
  cursor: pointer;
}

</style>
