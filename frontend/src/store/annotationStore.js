import api from "@/utilities/network";

const annotationStoreModule = {
	namespaced: true,
	state: () => ({
		//ner
		selectionStart: -1,
		selectionEnd: -1,

		//re

	}),
	getters: {
		getNERSelection(state) {
			return {
				selectionStart: state.selectionStart,
				selectionEnd: state.selectionEnd,
			}
		}
	},
	mutations: {},
	actions: {
		setNERSelection({commit, state, rootState}, payload) {
			// console.log("ner selection received", payload)
			state.selectionStart = payload.selectionStart;
			state.selectionEnd = payload.selectionEnd;
			// console.log("state", state)
		}
	}
}

export default annotationStoreModule;
