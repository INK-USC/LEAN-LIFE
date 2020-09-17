/* eslint-disable quotes */
/* eslint-disable indent */
import Vue from "vue";
import Vuex from "vuex";
 
Vue.use(Vuex);



const store = new Vuex.Store({
    state: {
        project: {
            task: null,
            explanation: null,
        },

        taskMap: {
            1: "SA",
            2: "NER",
            3: "RE",
        },

        explanationMap: {
            1: "None",
            2: "NL",
            3: "Trigger",
        },
    },
    getters: {
        projectType: state => state.taskMap[state.project.task],
        explanationType: state => state.explanationMap[state.project.explanation],
    },
    mutations: {
        setProjectInfo(state, payload) {
            state.project.task = payload.task;
            state.project.explanation = payload.explanation;
        },
    },
    actions: {}
   });

export default store;
