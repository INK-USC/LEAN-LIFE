/* eslint-disable no-undef */
/* eslint-disable no-console */
/* eslint-disable quotes */
import axios from 'axios';
import Vue from "vue";
import BASE_HTTP from "./baseHttp";


const vm = new Vue({
  el: "#upload-file",
  delimiters: ["${", "}"],
  data: {
    fileName: "",
    dataset: null,
    format: "json",
    disable: true,
    button_name: "Upload dataset",
    isNewlyCreated: false,
    projectType: null,
    errorMessage: "",
    intro: "",
    title: "",
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
        this.projectType = response.data["task"];
      });
    } else {
      this.projectType = JSON.parse(window.localStorage.getItem("project"))["task"];
    }

    if (JSON.parse(window.localStorage.getItem("newly_created")) === true) {
      document.getElementById("left_nav").style.pointerEvents = "none";
      document.getElementById("top_nav").style.pointerEvents = "none";
      this.isNewlyCreated = true;
    }
    
    const projectUrl = window.location.href.split('/').slice(3, 5).join("/");
    axios.get(`/api/${projectUrl}/heading_status`).then((response) => {
      if (response.data.has_documents) {
        this.intro = "If you'd like to add to your current set of uploaded documents, follow the instructions below.";
        this.title = "Add to your corpus below";
      } else {
        this.intro = "In order to start the annotation process, a corpus <u>must</u> be uploaded";
        this.title = "Import your corpus below";
      }
    });
  },

  methods: {
    reset() {
      this.fileName = "";
      this.dataset = null;
      this.format = "json";
      this.button_name = "Upload dataset";
    },
    updateFileName() {
      if (this.$refs.file.files.length) {
        this.fileName = this.$refs.file.files[0].name;
        // eslint-disable-next-line prefer-destructuring
        this.dataset = this.$refs.file.files[0];
        this.disable = false;
      } else {
        console.log("strange");
      }
    },
    uploadFile() {
      this.button_name = "Uploading...";
      this.disable = true;
      const formData = new FormData();
      formData.append("format", this.format);
      formData.append("dataset", this.dataset);
      const project = JSON.parse(window.localStorage.getItem("project"));
      const explanationType = project["task"];
      if (explanationType === 3) {
        formData.append("upload_type", "ner");
      } else {
        formData.append("upload_type", "pd");
      }
      BASE_HTTP.post("docs/create", formData)
        .then((response) => {
          let pageURL = response.request.responseURL;
          if (JSON.parse(window.localStorage.getItem("newly_created")) === true) {
            pageURL = pageURL.replace('docs', 'labels');
          }
          window.location.replace(pageURL);
        })
        .catch((err) => {
          this.reset();
          if (err.response.status == "500") {
            this.errorMessage = "Sorry, the file you uploaded is not in the right format. Please try again.";
          }
        });
    },
  },
});
