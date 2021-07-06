// store information about explanations
const explanationStoreModule = {
	namespaced: true,
	state: () => ({
		explanationInfo: {
			annotationId: -1,
		},
		explanationPopupInfo: {dialogVisible: false}
	}),
	getters: {
		getExplanationPopupInfo(state) {
			return state.explanationPopupInfo
		},
		getSelectedLabelInfo(state, getters, rootState, rootGetters) {
			const curAnnotation = rootGetters["document/getCurDoc"].annotations.find(annotation => annotation.id === state.explanationInfo.annotationId);
			if (!curAnnotation) {
				return {}
			}
			const curLabel = rootState.label.labelInfo.labels.find(label => label.id === curAnnotation.label)
			return curLabel;
		},
		getAnnotationInfo(state, getters, rootState, rootGetters) {
			const curAnnotation = rootGetters["document/getCurDoc"].annotations.find(annotation => annotation.id === state.explanationInfo.annotationId);
			return curAnnotation;
		},
	},
	mutations: {},
	actions: {
		showExplanationPopup({commit, dispatch, state, rootState}, payload) {
			const annotationId = payload.annotationId;
			state.explanationInfo.annotationId = annotationId;
			state.explanationPopupInfo.dialogVisible = true;
		},
		hideExplanationPopup({commit, dispatch, state, rootState}, payload) {
			state.explanationPopupInfo.dialogVisible = false;
		}
	}
}
export default explanationStoreModule
