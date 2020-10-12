import Vue from "vue";
import Vuex, {createLogger} from "vuex";
import router from "@/router";

Vue.use(Vuex);

const store = new Vuex.Store({
	state: {
		userInfo: null, // store info like userid, used preferred name
		projectInfo: {} //store project info like project name, type description
	},
	mutations: {
		login(state, userInfo) {
			state.userInfo = userInfo
		},
		logout(state) {
			console.log("log out ")
			state.userInfo = null;
			router.push("/").then(r => r);
		},
		setProject(state, projectInfo) {
			state.projectInfo = projectInfo;
		}
	},
	getters: {
		getUserInfo: state => {
			return state.userInfo;
		},
		getProjectInfo: state => {
			return state.projectInfo;
		}
	},
	actions: {},
	modules: {},
	plugins: [createLogger()]

});
export default store;
