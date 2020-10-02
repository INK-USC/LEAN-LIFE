import Vue from "vue";
import Vuex from "vuex";

Vue.use(Vuex);
import {createLogger} from 'vuex'

const store = new Vuex.Store({
	state: {
		userInfo: {name: "jim"}, // store info like userid, used preferred name
		projectInfo: {} //store project info like project name, type description
	},
	mutations: {
		login(state, userInfo) {
			state.userInfo = userInfo
		},
		logout(state) {
			// console.log("log out ")
			state.userInfo = undefined;
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
