/* eslint-disable no-undef */
/* eslint-disable no-console */
/* eslint-disable quotes */
import axios from 'axios';
import Vue from "vue";


const vm = new Vue({
  el: "#download-file",
  delimiters: ["${", "}"],
  data: {
    projectType: null,
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
  },
});
