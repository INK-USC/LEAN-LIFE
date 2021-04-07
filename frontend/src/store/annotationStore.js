import api from "@/utilities/network";
// store information about annotations for ner and re.
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
				clickedChunks: []
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
		},
		getREClickedChunks(state) {
			return state.annotationInfo.re.clickedChunks;
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
		},
		resetNERSelection({state}) {
			state.annotationInfo.ner.selectionStart = -1;
			state.annotationInfo.ner.selectionEnd = -1;
		},
		resetRESelection({state}) {
			state.annotationInfo.re.objStart = -1;
			state.annotationInfo.re.objEnd = -1;
			state.annotationInfo.re.sbjStart = -1;
			state.annotationInfo.re.sbjEnd = -1;
			state.annotationInfo.re.objText = "";
			state.annotationInfo.re.sbjText = "";
			state.annotationInfo.re.clickedChunks = [];
		},

		addREClickedChunks({state}, payload) {
			state.annotationInfo.re.clickedChunks = [...state.annotationInfo.re.clickedChunks, payload.chunk];
		}
	}
}

export default annotationStoreModule;
