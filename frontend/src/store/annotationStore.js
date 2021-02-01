import api from "@/utilities/network";

const annotationStoreModule = {
	namespaced: true,
	state: () => ({
		annotationInfo: {
			//ner
			selectionStart: -1,
			selectionEnd: -1,

			//re
		}
	}),
	getters: {
		getNERSelection(state) {
			return {
				selectionStart: state.annotationInfo.selectionStart,
				selectionEnd: state.annotationInfo.selectionEnd,
			}
		}
	},
	mutations: {},
	actions: {
		setNERSelection({commit, state, rootState}, payload) {
			// console.log("ner selection received", payload)
			state.annotationInfo.selectionStart = payload.selectionStart;
			state.annotationInfo.selectionEnd = payload.selectionEnd;
			// console.log("state", state)
		}
	}
}

export default annotationStoreModule;
