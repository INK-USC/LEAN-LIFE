import Vue from "vue";
import VueRouter from "vue-router";
import Home from "../views/Home.vue";
import Login from "@/components/Login";
import UploadFile from "@/components/createProject/UploadFile";
import Projects from "@/components/Projects";
import Logout from "@/components/Logout";

Vue.use(VueRouter);

const routes = [
	{
		path: "/",
		name: "Home",
		component: Home
	},
	// {
	// 	path: "/about",
	// 	name: "About",
	// 	// route level code-splitting
	// 	// this generates a separate chunk (about.[hash].js) for this route
	// 	// which is lazy-loaded when the route is visited.
	// 	component: () =>
	// 			import(/* webpackChunkName: "about" */ "../views/About.vue")
	// },
	{
		path: "/projects",
		name: "Projects",
		component: Projects,
	},
	{
		path: "/login",
		name: "Login",
		component: Login
	},
	{path: "/logout", component: Logout},
	{path: "/create/update", name: "CreateProject", component: UploadFile}
];

const router = new VueRouter({
	mode: "history",
	base: process.env.BASE_URL,
	routes,
	linkActiveClass: "is-active"
});

export default router;
