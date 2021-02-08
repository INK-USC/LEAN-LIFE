import api from "@/utilities/network";

const annotationStoreModule = {
	namespaced: true,
	state: () => ({
		annotationInfo: {
			ner: {
				selectionStart: -1,
				selectionEnd: -1,
			},
			re: {
				objStart: -1,
				objEnd: -1,
				sbjStart: -1,
				sbjEnd: -1,
				objText: "",
				sbjText: "",
			}
		}
	}),
	getters: {
		getNERSelection(state) {
			return {
				selectionStart: state.annotationInfo.ner.selectionStart,
				selectionEnd: state.annotationInfo.ner.selectionEnd,
			}
		},
		getRESelection(state) {
			return {
				obj_start_offset: state.annotationInfo.re.objStart,
				obj_end_offset: state.annotationInfo.re.objEnd,
				sbj_start_offset: state.annotationInfo.re.sbjStart,
				sbj_end_offset: state.annotationInfo.re.sbjEnd,
				obj_text: state.annotationInfo.re.objText,
				sbj_text: state.annotationInfo.re.sbjText,
			}
		}
	},
	mutations: {},
	actions: {
		setNERSelection({commit, state, rootState}, payload) {
			// console.log("ner selection received", payload)
			state.annotationInfo.ner.selectionStart = payload.selectionStart;
			state.annotationInfo.ner.selectionEnd = payload.selectionEnd;
			// console.log("state", state)
		},
		setRESelection({commit, state, rootState}, payload) {
			state.annotationInfo.re.objStart = payload.objStart;
			state.annotationInfo.re.objEnd = payload.objEnd;
			state.annotationInfo.re.sbjStart = payload.sbjStart;
			state.annotationInfo.re.sbjEnd = payload.sbjEnd;
			state.annotationInfo.re.objText = payload.objText;
			state.annotationInfo.re.sbjText = payload.sbjText;
		}
	}
}

export default annotationStoreModule;
