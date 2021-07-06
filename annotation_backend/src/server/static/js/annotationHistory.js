import axios from 'axios';
import Vue from "vue";
axios.defaults.xsrfCookieName = 'csrftoken';
axios.defaults.xsrfHeaderName = 'X-CSRFToken';
import HTTP from "./http";

const vm = new Vue({
  el: "#annotation-history",
  delimiters: ["${", "}"],
  data: {
    fileName: "",
    annotations: null,
    projectType: null,
    currentPageHistory: [],
    nextPageUrl: null,
    prevPageUrl: null,
    currentLabels: {},
    canUpload: false,
    isNewlyCreated: false,
    leadUserToAnnotation: false,
    errorMessage: "",
  },

  created() {
    let update = false;
    let endPoint = '';

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
        this.projectType = response.data["task"];
        if (this.projectType === 2) {
          endPoint = 'history/ner/';
        } else {
          endPoint = 'history/re/';
        }
    
        HTTP.get(endPoint).then((response) => {
          const responseData = response.data;
          this.nextPageUrl = responseData.next;
          this.prevPageUrl = responseData.previous;
          this.currentPageHistory = responseData.results;
        });
      });
    } else {
      this.projectType = JSON.parse(window.localStorage.getItem("project"))["task"];
      if (this.projectType === 2) {
        endPoint = 'history/ner/';
      } else {
        endPoint = 'history/re/';
      }
  
      HTTP.get(endPoint).then((response) => {
        const responseData = response.data;
        this.nextPageUrl = responseData.next;
        this.prevPageUrl = responseData.previous;
        this.currentPageHistory = responseData.results;
      });
    }
    
    if (JSON.parse(window.localStorage.getItem("newly_created")) === true) {
      this.isNewlyCreated = true;
    }

    HTTP.get("labels").then((response) => {
      for (let i=0; i < response.data.length; i++) {
        this.currentLabels[response.data[i].id] = response.data[i].text;
      }
    });
  },

  methods: {
    deleteAnnotation(historyId, index) {
      let url = '';
      if (this.projectType === 2) {
        url = `history/ner/${historyId}`;
      } else {
        url = `history/re/${historyId}`;
      }
      HTTP.delete(url).then((response) => {
        this.currentPageHistory.splice(index);
      });
    },

    updateFileName() {
      if (this.$refs.file.files.length) {
        this.fileName = this.$refs.file.files[0].name;
        // eslint-disable-next-line prefer-destructuring
        this.annotations = this.$refs.file.files[0];
        this.canUpload = true;      }
    },

    uploadFile(action) {
      if (this.annotations) {
        const formData = new FormData();
        formData.append("history", this.annotations);
        formData.append("action", action);
        formData.append("task", this.projectType);
        HTTP.post(`history/seed/`, formData).then((response) => {
          this.fileName = "";
          this.annotations = null;
          document.getElementById('file-upload').value = "";
          this.canUpload = false;
          let endPoint = "";
          if (this.projectType === 2) {
            endPoint = 'history/ner/';
          } else {
            endPoint = 'history/re/';
          }
      
          HTTP.get(endPoint).then((response) => {
            const responseData = response.data;
            this.nextPageUrl = responseData.next;
            this.prevPageUrl = responseData.previous;
            this.currentPageHistory = responseData.results;
            if (this.isNewlyCreated) {
              this.leadUserToAnnotation = true;
            }
          });
        }).catch((err) => {
          console.log("here")
          if (err.response.status == "500") {
            this.fileName = "";
            this.annotations = null;
            document.getElementById('file-upload').value = "";
            this.canUpload = false;
            this.errorMessage = "Sorry, the file you uploaded is not in the right format. Please look above, for the correct format.";
          }
        });
      }
    },

    nextPage() {
      if (this.nextPageUrl) {
        HTTP.get(this.nextPageUrl).then((response) => {
          const responseData = response.data;
          this.nextPageUrl = responseData.next;
          this.prevPageUrl = responseData.previous;
          this.currentPageHistory = responseData.results;
        });
      }
    },

    previousPage() {
      if (this.prevPageUrl) {
        HTTP.get(this.prevPageUrl).then((response) => {
          const responseData = response.data;
          this.nextPageUrl = responseData.next;
          this.prevPageUrl = responseData.previous;
          this.currentPageHistory = responseData.results;
        });
      }
    },

    goToAnnotations() {
      const nextUrl = window.location.href.split('/').slice(0, 5).join("/");
      window.location.replace(nextUrl);
    }
  },
});
