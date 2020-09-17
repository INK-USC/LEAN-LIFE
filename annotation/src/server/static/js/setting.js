import axios from 'axios';
import Vue from 'vue';
import HTTP from './http';


const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  data: {
    nounChunk: null,
    modelBackedRecs: null,
    history: null,
    isOnlineOn: null,
    encodingStrategy: null,
    batch: null,
    epoch: null,
    activeLearningStrategy: null,
    acquire: null,
    projectType: null,
    newlyCreated: false,
    prompt: false,
  },
  methods: {
    save(url = '') {
      const payload = {
      };
      if (this.nounChunk != null) {
        payload.noun_chunk = this.nounChunk;
      }
      if (this.modelBackedRecs != null) {
        payload.model_backed_recs = this.modelBackedRecs;
      }
      if (this.history != null) {
        payload.history = this.history;
      }
      if (this.isOnlineOn != null) {
        payload.is_online_on = this.isOnlineOn;
      }
      if (this.encodingStrategy != null) {
        payload.encoding_strategy = this.encodingStrategy;
      }
      if (this.batch != null) {
        payload.batch = this.batch;
      }
      if (this.epoch != null) {
        payload.epoch = this.epoch;
      }
      if (this.activeLearningStrategy != null) {
        payload.active_learning_strategy = this.activeLearningStrategy;
      }
      if (this.acquire != null) {
        payload.acquire = this.acquire;
      }
      HTTP.put(`settings/`, payload).then((response) => {
      });
      let nextUrl = '';
      if (url.length === 0) {
        nextUrl = window.location.href.split('/').slice(0, 5).join("/");
      } else {
        nextUrl = url;
      }
      window.location.replace(nextUrl);
    },
    reset() {
      this.nounChunk = false;
      this.modelBackedRecs = false;
      this.history = false;
      this.isOnlineOn = false;
      this.encodingStrategy = 1;
      this.batch = 10;
      this.epoch = 5;
      this.activeLearningStrategy = 1;
      this.acquire = 5;
    },
  },
  watch: {
  },
  created() {
    HTTP.get('settings').then((response) => {
      this.nounChunk = response.data.noun_chunk;
      this.modelBackedRecs = response.data.model_backed_recs;
      this.history = response.data.history;
      this.isOnlineOn = response.data.is_online_on;
      this.encodingStrategy = response.data.encoding_strategy;
      this.batch = response.data.batch;
      this.epoch = response.data.epoch;
      this.activeLearningStrategy = response.data.active_learning_strategy;
      this.acquire = response.data.acquire;
    });
    
    let update = false;

    if (window.localStorage.getItem('project') === null) {
      update = true;
    } else if (JSON.parse(window.localStorage.getItem("project"))["id"] === null && JSON.parse(window.localStorage.getItem("newly_created")) === true) {
      const project = JSON.parse(localStorage.getItem("project"));
      project.id = window.location.href.split('/')[4];
      localStorage.setItem('project', JSON.stringify(project))
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
      this.newlyCreated = true;
    }
  },
});
