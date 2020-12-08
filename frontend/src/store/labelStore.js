import api from "@/utilities/network";

const labelStoreModule = {
	namespaced: true,
	state: () => ({
		labelInfo: {
			labels: []
		}
	}),
	getters: {
		getLabels(state) {
			return state.labelInfo.labels
		}
	},
	mutations: {},
	actions: {
		fetchLabels({commit, state, rootState}, payload) {
			api
				.get(`/projects/${rootState.projectInfo.id}/labels/`)
				.then(res => {
					state.labelInfo.labels = res;
					return res
				})
		}
	}

}

export default labelStoreModule
