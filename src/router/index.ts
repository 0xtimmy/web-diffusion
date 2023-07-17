import { createRouter, createWebHistory, RouteRecordRaw } from "vue-router";
import Tests from "../views/Tests.vue";
import Diffuser from "../views/Diffuser.vue";
import Performance from "../views/Performance.vue"

const routes: Array<RouteRecordRaw> = [
  {
    path: "/",
    name: "Diffuser",
    component: Diffuser
  },
  {
    path: "/performance",
    name: "Performance",
    component: Performance
  },
  {
    path: "/tests",
    name: "Tests",
    component: Tests,
  }
  
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
