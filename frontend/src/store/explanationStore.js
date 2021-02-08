import api from "@/utilities/network";

const explanationStoreModule = {
	namespaced: true,
	state: () => ({
		explanationInfo: {
			selectedLabelInfo: {},
			annotationInfo: {},
		},
		explanationPopupInfo: {dialogVisible: false}
	}),
	getters: {
		getExplanationPopupInfo(state) {
			return state.explanationPopupInfo
		},
		getSelectedLabelInfo(state) {
			return state.explanationInfo.selectedLabelInfo;
		},
		getAnnotationInfo(state) {
			return state.explanationInfo.annotationInfo;
		}
	},
	mutations: {},
	actions: {
		showExplanationPopup({commit, dispatch, state, rootState}, payload) {
			console.log("payload", payload)
			console.log("root state", rootState)

			state.explanationInfo.selectedLabelInfo = payload.label;
			state.explanationInfo.annotationInfo = payload.annotation;
			state.explanationPopupInfo.dialogVisible = true;
			console.log("store", state.explanationInfo)
		},
		hideExplanationPopup({commit, dispatch, state, rootState}, payload) {
			state.explanationPopupInfo.dialogVisible = false;
			state.explanationInfo.selectedLabel = {}
		}
	}
}
export default explanationStoreModule
