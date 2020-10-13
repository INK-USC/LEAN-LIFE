import Vue from "vue";
import Vuex, {createLogger} from "vuex";
import router from "@/router";
import createPersistedState from "vuex-persistedstate";
import api from "@/utilities/network";

Vue.use(Vuex, api);

const store = new Vuex.Store({
	state: {
		userInfo: null, // store info like userid, used preferred name
		projectInfo: {},//store project info like project name, type description
		isLoading: false,
	},
	mutations: {
		login(state, loginCredential) {
			api.post("login", {}).then(() => {
//TODO extract from login into here
			})
			state.userInfo = loginCredential;
			router.push({name: "Projects"}).then(r => r);
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
	plugins: [createLogger(), createPersistedState()]

});
export default store;
