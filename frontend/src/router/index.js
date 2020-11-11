import Vue from "vue";
import VueRouter from "vue-router";
import Home from "../views/Home.vue";
import Login from "@/components/Login";
import UploadDocument from "@/components/project/UploadDocument";
import Projects from "@/components/Projects";
import Logout from "@/components/Logout";
import store from "@/store";
import Project from "@/views/Project";
import CreateProjectModal from "@/components/CreateProjectModal";

Vue.use(VueRouter);

const AuthGuard = (to, from, next) => {
	let isAuthenticated = false;
	if (store.getters.getUserInfo) {
		isAuthenticated = true;
	} else {
		isAuthenticated = false;
	}
	if (isAuthenticated) {
		next();
	} else {
		next("/login");
	}
}

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
		beforeEnter: AuthGuard,
		component: Projects,
	},
	{
		path: "/login",
		name: "Login",
		component: Login
	},
	{path: "/logout", component: Logout},
	{
		path: "/project/", component: Project,
		children: [
			{path: "edit", name: "CreateProject", component: CreateProjectModal},
			{path: "upload", name: "UploadFile", component: UploadDocument}
		]
	},
];

const router = new VueRouter({
	mode: "history",
	base: process.env.BASE_URL,
	routes,
	linkActiveClass: "is-active"
});

router.afterEach(() => {
	if (store.getters.getUserInfo) {
		store.dispatch('inspectToken').then(_ => _)
	}
})

export default router;
