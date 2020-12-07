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
			// console.log("fetch doc", state.documentInfo, rootState)
			api
				.get(`/projects/${rootState.projectInfo.id}/labels/`)
				.then(res => {
					state.labelInfo.labels = res;
					// state.documentInfo.documents = res.results;
					// state.documentInfo.totalDocumentCount = res.count;
					return res
				})
		}
	}

}

export default labelStoreModule
