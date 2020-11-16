/* eslint-disable key-spacing */
import axios from "axios";
import Vue from "vue";
import { AutoComplete, Modal, Form, Icon, Button, Radio } from "ant-design-vue";
import HTTP from "./http";


import "ant-design-vue/dist/antd.css";
import AnnotationDocument from "./utils";

axios.defaults.xsrfCookieName = "csrftoken";
axios.defaults.xsrfHeaderName = "X-CSRFToken";

Vue.use(Modal);
Vue.use(Form);
Vue.use(Icon);
Vue.use(AutoComplete);
Vue.use(Button);
Vue.use(Radio);

const annotationMixin = {
  data() {
    return {
      pageNumber: 0,
      annotationDocs: {},
      labels: [],
      triggerLabels: [],
      guideline: "",
      searchQuery: "",
      numOfAnnotatedDocs: 0,
      nextURL: "",
      offset: 0,
      typeOfDoc: "all",
      totalDocCount: 0,
      isLoading: true,
      loadingMsg: "Loading...",
      settings: {},
      server: false,
      serverMsg: "OFF",
      explanationPopup: false,
      lastAnnotationId: null,
      nextDocQuery: null,
      prevDocQuery: null,
      projectType: null,
      explanationType: null,
      documentsProcessed: false,
      selectedWords: [],
      isNewlyCreated: 0,
      refresh: false,

    };
  },

  methods: {
    getOffsetFromUrl(url) {
      const offsetMatch = url.match(/[?#].*offset=(\d+)/);
      if (offsetMatch == null) {
        return 0;
      }
      return parseInt(offsetMatch[1], 10);
    },

    nextPage() {
      this.pageNumber += 1;
      if (this.pageNumber === Object.keys(this.annotationDocs).length) {
        if (this.nextDocQuery) {
          this.nextURL = this.nextDocQuery;
          this.getDocuments();
          this.pageNumber = 0;
        } else {
          // No more documents
          this.pageNumber = Object.keys(this.annotationDocs).length - 1;
        }
      }
    },

    prevPage() {
      this.pageNumber -= 1;
      if (this.pageNumber === -1) {
        if (this.prevDocQuery) {
          this.nextURL = this.prevDocQuery;
          this.getDocuments();
          this.pageNumber = this.settings.acquire - 1;
        } else {
          // No more documents
          this.pageNumber = 0;
        }
      }
    },

    updateAnnnotationStatus(isAnnotated) {
      const docId = this.annotationDocs[this.pageNumber].id;
      HTTP.patch(`docs/${docId}`, { annotated: isAnnotated }).then((response) => {});
    },

    noAnnotations() {
      // Probably shouldn't allow this button to be clicked if previously anntotated
      const check = this.annotationDocs[this.pageNumber].annotated;
      if (!check) {
        this.updateAnnnotationStatus(true);
        this.annotationDocs[this.pageNumber].annotated = true;
        this.numOfAnnotatedDocs += 1;
      }
      this.nextPage();
    },

    // eslint-disable-next-line space-before-function-paren
    async process_data(response) {
      this.annotationDocs = {};
      this.isLoading = true;
      this.loadingMsg = "Preparing Documents for annotations";
      if (response.data.results.length > 0) {
        let annotated_doc_count= 0
        for (let i=0; i < response.data.results.length; i++) {
          if(response.data.results[i].annotated){
            annotated_doc_count++;
          }
          this.annotationDocs[i] = new AnnotationDocument(i, response.data.results[i], this.explanationType, this.projectType);
        }
        if(annotated_doc_count>2){
            document.getElementById("train_modal_btn").disabled=false
        }else{
            document.getElementById("train_modal_btn").disabled='disabled'
        }
        this.totalDocCount = response.data.count;
        this.nextDocQuery = response.data.next;
        this.prevDocQuery = response.data.previous;
        await HTTP.get(`docs/${this.annotationDocs[this.pageNumber].id}/recommendations/${this.projectType}/`).then(
          (recomResponse) => {
            const rec = recomResponse.data.recommendation;
            this.annotationDocs[this.pageNumber].addRecommendations(rec);
            this.isLoading = false;
          },
        );

        for (let i = 0; i < Object.keys(this.annotationDocs).length; i++) {
          if (i !== this.pageNumber) {
            HTTP.get(`docs/${this.annotationDocs[i].id}/recommendations/${this.projectType}/`).then(
              (recomResponse) => {
                const rec = recomResponse.data.recommendation;
                this.annotationDocs[i].addRecommendations(rec);
              },
            );
          }
        }
        this.offset = this.getOffsetFromUrl(this.nextURL);
        this.documentsProcessed = true;
      } else {
        this.loadingMsg = "Sorry no further documents were found for this project.";
        window.location.href = `projects/${window.localStorage.getItem("project").id}/docs/`
      }
    },

    async getDocuments() {
      HTTP.get(this.nextURL).then((response) => this.process_data(response));
    },

    async searchDocs() {
      this.nextURL = `docs/?q=${this.searchQuery}`;
      await this.getDocuments();
    },

    async submit() {
      this.nextURL = `docs/?offset=${this.offset}&limit=${this.settings.acquire}`;
      await this.getDocuments();
      this.pageNumber = 0;
    },
    
    replaceNull(shortcut) {
      if (shortcut === null) {
        shortcut = "";
      }
      shortcut = shortcut.split(" ");
      return shortcut;
    },

    refreshTriggers(newTriggers) {
      this.annotationDocs[this.pageNumber].triggers[this.lastAnnotationId] = newTriggers;
    },

    refreshNlExplanations(newExplanations) {
      this.annotationDocs[this.pageNumber].nlExplanations[this.lastAnnotationId] = newExplanations;
    },

    trainModel(){
        HTTP.post("mock", {}).then(res=>{})
    },
  },

  watch: {
    typeOfDoc() {
      console.log(this.typeOfDoc);
      this.submit();
    },
    annotations() {
      HTTP.get("progress").then((response) => {
        this.numOfAnnotatedDocs = response.data.annotated_doc_count;
      });
    },
    // offset() {
    //   this.storeOffsetInUrl(this.offset);
    // },
  },


  created() {
    let update = false;

    if (window.localStorage.getItem('project') === null) {
      update = true;
    } else if (JSON.parse(window.localStorage.getItem("project"))["id"] === null && JSON.parse(window.localStorage.getItem("newly_created")) === true) {
      const project = JSON.parse(localStorage.getItem("project"));
      project.id = window.location.href.split('/')[4];
      localStorage.setItem('project', JSON.stringify(project));
    } else if (JSON.parse(window.localStorage.getItem("project"))["id"] !== null && JSON.parse(window.localStorage.getItem("project"))["id"] !== window.location.href.split('/')[4]) {
      update = true;
    }
    
    if (update) {
      const projectUrl = window.location.href.split('/').slice(3, 5).join("/");
      axios.get(`/api/${projectUrl}`).then((response) => {
        window.localStorage.setItem("project", JSON.stringify(response.data));
        this.projectType = response.data.task;
        this.explanationType = response.data.explanation_type;
        update = false;
      });
    } else {
      const project = JSON.parse(localStorage.getItem("project"));
      this.projectType = project.task;
      this.explanationType = project.explanation_type;
    }

    if (JSON.parse(window.localStorage.getItem("newly_created")) === true) {
      this.isNewlyCreated = 1;
      window.localStorage.removeItem("newly_created");
    }
     
    HTTP.get("labels").then((response) => {
      this.labels = response.data;
    });

    // Color codes for the trigger labels.
    const triggerBackgroundColors = ["#95de64", "#5cdbd3", "#85a5ff", "#b37feb"];

    for (let i = 0; i < 4; i++) {
      const tlabel = {
        id: i,
        text: i,
        text_color: "white",
        background_color: triggerBackgroundColors[i],
        text_decoration: "",
        shortcut: i,
      };
      this.triggerLabels.push(tlabel);
    }

    HTTP.get("settings")
      .then((response) => {
        this.settings.nounChunk = response.data.noun_chunk;
        this.settings.modelBackedRecs = response.data.model_backed_recs;
        this.settings.history = response.data.history;
        this.settings.isOnlineOn = response.data.is_online_on;
        this.settings.encodingStrategy = response.data.encoding_strategy;
        this.settings.batch = response.data.batch;
        this.settings.epoch = response.data.epoch;
        this.settings.activeLearningStrategy = response.data.active_learning_strategy;
        this.settings.acquire = response.data.acquire;
        // this.connectServer().then(response => {
        //   if (this.online === true && this.server === true) {
        //     this.initiatelearning().then(response => {
        //       console.log('model server on')
        //       this.submit()
        //     })
        //   } else {
        //     console.log('model server off')
        //     this.submit()
        //   }
        // })
        this.submit();
      });
    HTTP.get().then((response) => {
      this.guideline = response.data.guideline;
    });
  },

  computed: {
    achievement() {
      const percentage = Math.round(
        (this.numOfAnnotatedDocs / this.totalDocCount) * 100
      );
      return this.totalDocCount > 0 ? percentage : 0;
    },

    id2label() {
      const id2label = {};
      for (let i = 0; i < this.labels.length; i++) {
        const label = this.labels[i];
        id2label[label.id] = label;
      }
      return id2label;
    },

    progressColor() {
      if (this.achievement < 30) {
        return "is-danger";
      }
      if (this.achievement < 70) {
        return "is-warning";
      }
      return "is-primary";
    },

    serverOn() {
      if (this.server === true) {
        this.serverMsg = "ON";
        return "is-primary";
      }
      this.serverMsg = "OFF";
      return "is-danger";
    },
    
    onlineOn() {
      if (this.settings.modelBackedRecs === true && this.server === false) {
        return "is-danger";
      }
      return "is-primary";
    },

    // getTutorialHeading() {
    //   if (this.projectType === 1) {
    //     return "SA Annotation Guideline";
    //   } else if (this.projectType === 2) {
    //     return "NER Annotation Guideline";
    //   } else if (this.projectType === 3) {
    //     return "RE Annotation Guideline";
    //   }
    // },
  },
};

export default annotationMixin;
