import Vue from "vue";
import Vuex, {createLogger} from "vuex";
import router from "@/router";
import createPersistedState from "vuex-persistedstate";
import api from "@/utilities/network";
import jwt_decode from "jwt-decode";

Vue.use(Vuex, api);
//if need to use axios, use api.get/post

const store = new Vuex.Store({
	state: {
		userInfo: null, // store info like userid, used preferred name
		projectInfo: {},//store project info like project name, type description
		createEditProjectRelatedInfo: {
			step: 0,
		},
	},
	mutations: {
		login(state, loginCredential) {
			state.userInfo = loginCredential;
			api.defaults.headers['Authorization'] = `JWT ${loginCredential.token}`
			router.push({name: "Projects"}).then(r => r);
		},
		logout(state) {
			console.log("log out ")
			state.userInfo = null;
			router.push("/").then(r => r);
		},
		updateAxios(state) {
			if (this.state.userInfo) {
				api.defaults.headers['Authorization'] = `JWT ${this.state.userInfo.token}`
			}
		},
		updateToken(state, token) {
			state.userInfo.token = token;
			api.defaults.headers['Authorization'] = `JWT ${token}`
		},
		setProject(state, projectInfo) {
			state.projectInfo = projectInfo;
		},
		updateProjectEditingStep(state, payload) {
			state.createEditProjectRelatedInfo.step = payload.step
		}
	},
	getters: {
		getUserInfo: state => {
			return state.userInfo;
		},
		getProjectInfo: state => {
			return state.projectInfo;
		},
		getProjectCreatingStep: state => {
			return state.createEditProjectRelatedInfo.step;
		},
		getEmptyProject: state => {
			return {
				name: "",
				description: "",
				guideline: "test",
				task: "",
				users: []
			}
		}
	},
	actions: {
		refreshToken() {
			api.post('/auth/refresh_token/', {token: this.state.userInfo.token}).then(res => {
				this.commit("updateToken", res.token)
			})
		},
		inspectToken() {
			// 1. IF it has expired => DO NOT REFRESH / PROMPT TO RE-OBTAIN TOKEN
			// 2. IF it is expiring in 30 minutes (1800 second) AND it is not reaching its lifespan (7 days — 30 mins = 630000–1800 = 628200) => REFRESH
			// 3. IF it is expiring in 30 minutes AND it is reaching its lifespan => DO NOT REFRESH

			const token = this.state.userInfo.token;
			if (token) {
				const decoded = jwt_decode(token);
				const exp = decoded.exp;
				const orig_iat = decoded.orig_iat;
				if (exp < Date.now() / 1000) {
					this.commit("logout")
				} else if (exp - (Date.now() / 1000) < 1800 && (Date.now() / 1000) - orig_iat < 628200) {
					this.dispatch('refreshToken')
				} else if (exp - (Date.now() / 1000) < 1800) {
					// DO NOTHING, DO NOT REFRESH
				}
			}
		}
	},
	modules: {},
	plugins: [createLogger(),
		createPersistedState({
			storage: window.sessionStorage,
		})
	]

});
export default store;
