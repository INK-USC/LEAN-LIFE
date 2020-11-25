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
import DocumentList from "@/components/project/overview/DocumentList";
import Document from "@/views/Document";
import LabelCreation from "@/components/project/overview/LabelCreation";

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

			{
				path: "doc/", component: Document,
				children: [
					{path: "upload", name: "UploadFile", component: UploadDocument},
					{path: "list", name: "DocumentList", component: DocumentList},
				]
			},
			{path: "labels", name: "LabelCreationSpace", component: LabelCreation}
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
