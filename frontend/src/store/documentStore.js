import api from "@/utilities/network";
import {formatAnnotations, formatExplanations} from "@/store/util";
// store information about document
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
			return api
				.get(`/projects/${rootState.projectInfo.id}/docs/?page=${state.documentInfo.curPage}&page_size=${state.documentInfo.pageSize}`)
				.then(res => {
					state.documentInfo.documents = res.results.results;
					state.documentInfo.documents.forEach(doc => {
						doc.formattedAnnotations = formatAnnotations(doc, rootState.projectInfo.task)
						doc.annotations = doc.annotations.sort((a, b) => a.id - b.id)
					})
					if (rootState.projectInfo.explanation_type === 3) {
						//prep for trigger explanation
						state.documentInfo.documents.forEach(doc => {
							doc.annotations.forEach(annotation => {
								annotation.formattedTriggerExplanation = formatExplanations(annotation)
							});
						})
					}
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
