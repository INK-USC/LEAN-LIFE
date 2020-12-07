import api from "@/utilities/network";

const documentStoreModule = {
	namespaced: true,
	state: () => ({
		documentInfo: {
			documents: [],
			curPage: 2,
			totalDocumentCount: 0,
			curDocIndex: 100,
		}
	}),
	getters: {
		getDocuments(state) {
			return state.documentInfo
		}
	},
	mutations: {},
	actions: {
		fetchDocuments({commit, state, rootState}, payload) {
			// console.log("fetch doc", state.documentInfo, rootState)
			api
				.get(`/projects/${rootState.projectInfo.id}/docs/?page=${state.documentInfo.curPage}`)
				.then(res => {
					state.documentInfo.documents = res.results;
					state.documentInfo.totalDocumentCount = res.count;
					return res
				})
		}
	}

}

export default documentStoreModule
