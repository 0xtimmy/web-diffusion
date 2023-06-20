import { createRouter, createWebHistory, RouteRecordRaw } from "vue-router";
import Tests from "../views/Tests.vue";
import Diffuser from "../views/Diffuser.vue";

const routes: Array<RouteRecordRaw> = [
  {
    path: "/",
    name: "Tests",
    component: Tests,
  },
  {
    path: "/diffuser",
    name: "Diffuser",
    component: Diffuser
  }
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
