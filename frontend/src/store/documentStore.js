import api from "@/utilities/network";

const documentStoreModule = {
	namespaced: true,
	state: () => ({
		documentInfo: {
			documents: [],
			curPage: 1,
			maxPage: 1,
			totalDocCount: 0,
			curDocIndex: 0,
			annotatedDocCount: 0,
			pageSize: 10
		}
	}),
	getters: {
		getDocuments(state) {
			return state.documentInfo
		},
		getCurDoc(state) {
			return state.documentInfo.documents[state.documentInfo.curDocIndex]
		}
	},
	mutations: {
		setCurDocIndex(state, curDocIndex) {
			state.documentInfo.curDocIndex = curDocIndex;
		}
	},
	actions: {
		fetchDocuments({commit, state, rootState}, payload) {
			api
				.get(`/projects/${rootState.projectInfo.id}/docs/?page=${state.documentInfo.curPage}&page_size=${state.documentInfo.pageSize}`)
				.then(res => {
					state.documentInfo.documents = res.results.results;
					state.documentInfo.totalDocCount = res.count;
					state.documentInfo.annotatedDocCount = res.results.annotatedCount;
					state.documentInfo.maxPage = Math.ceil(state.documentInfo.totalDocCount / state.documentInfo.pageSize);
					return res
				})
		},
		updateCurPage({commit, dispatch, state, rootState}, payload) {
			console.log("page changed", payload)
			state.documentInfo.curPage = payload.newPage;
			state.documentInfo.curDocIndex = 0;
			dispatch('fetchDocuments');
		},
		updateCurDocIndex({commit, dispatch, state, rootState}, payload) {
			if (payload.curDocIndex >= 0 && payload.curDocIndex < state.documentInfo.documents.length) {
				state.documentInfo.curDocIndex = payload.curDocIndex;
			} else if (payload.curDocIndex >= state.documentInfo.documents.length) {
				dispatch('updateCurPage', {newPage: state.documentInfo.curPage + 1})
			} else if (payload.curDocIndex < 0) {
				dispatch('updateCurPage', {newPage: state.documentInfo.curPage - 1})
			}
		}
	}

}

export default documentStoreModule
