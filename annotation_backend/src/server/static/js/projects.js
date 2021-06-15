/* eslint-disable quotes */
import Vue from 'vue';
import axios from 'axios';

axios.defaults.xsrfCookieName = 'csrftoken';
axios.defaults.xsrfHeaderName = 'X-CSRFToken';
const baseUrl = window.location.href.split('/').slice(0, 3).join('/');

const vm = new Vue({
  el: '#projects_root',
  delimiters: ['[[', ']]'],
  data: {
    items: [],
    isCreate: false,
    isDelete: false,
    project: null,
    selected: 'All Project',
    isWarningUpdate: false,
    isExplChange: false,
    isTaskChange: false,
  },

  methods: {
    deleteProject() {
      axios.delete(`${baseUrl}/api/projects/${this.project.id}/`).then((response) => {
        this.isDelete = false;
        const index = this.items.indexOf(this.project);
        this.items.splice(index, 1);
      });
    },

    setFormProject() {
      window.localStorage.setItem("newly_created", true);

      const project_dict = {
        "name": "",
        "description": "",
        "users": "",
        "task": "",
        "explanation_type": ""
      };

      const project_name = document.getElementById('id_name').value;
      const project_desc = document.getElementById('id_description').value;
      const project_task = document.getElementById('id_task').value;
      const project_expl = document.getElementById('id_explanation_type').value;
      const project_user = document.getElementById('id_users').value;

      project_dict["name"] = project_name;
      project_dict["description"] = project_desc;
      project_dict["users"] = project_user;
      project_dict["task"] = project_task;
      project_dict["explanation_type"] = project_expl;

      window.localStorage.setItem("project", JSON.stringify(project_dict));
    },

    checkUpdate() {
      const projectTask = document.getElementById('update_task').value;
      const projectExpl = document.getElementById('update_explanation_type').value;

      const currProject = JSON.parse(window.localStorage.getItem("project"));

      if (parseInt(projectTask, 10) !== parseInt(currProject['task'], 10)) {
        this.isTaskChange = true;
      }

      if (parseInt(projectExpl, 10) !== parseInt(currProject['explanation_type'], 10)) {
        this.isExplChange = true;
      }

      if (this.isTaskChange || this.isExplChange) {
        this.isWarningUpdate = true;
        event.preventDefault();
      }
    },

    submitUpdateForm() {
      document.getElementById("update_form").submit();
    },

    doesKeyExist(key) {
      if (window.localStorage.getItem(key) === null) {
        return false;
      }
      return true;
    },

    setProject(project, toggle = 1) {
      this.project = project;
      if (toggle < 1) {
        this.isDelete = true;
      } else {
        window.localStorage.setItem("project", JSON.stringify(project));
        if (this.doesKeyExist("project")) {
          if (toggle === 2) {
            window.location.href = '/projects/' + project.id + '/update'
          }
        } else {
          while (true) {
            if (this.doesKeyExist('project')) {
              break;
            }
          }
          if (toggle === 2) {
            window.location.href = '/projects/' + project.id + '/update'
          }
        }
      }
    },

    matchType(projectType) {
      if (projectType === 'SentimentAnalysis') {
        return this.selected === 'Sentiment Analysis';
      }
      if (projectType === 'NamedEntityRecognition') {
        return this.selected === 'Named Entity Recognition';
      }
      if (projectType === 'RelationExtraction') {
        return this.selected === 'Relation Extraction';
      }
      return false;
    },

    getDaysAgo(dateStr) {
      const updatedAt = new Date(dateStr);
      const currentTm = new Date();

      // difference between days(ms)
      const msDiff = currentTm.getTime() - updatedAt.getTime();

      // convert daysDiff(ms) to daysDiff(day)
      const daysDiff = Math.floor(msDiff / (1000 * 60 * 60 * 24));

      return daysDiff;
    },
  },

  computed: {
    selectedProjects() {
      return this.items.filter(item => this.selected === 'All Project' || this.matchType(item.project_type));
    },
    isUpdate() {
      if (window.location.href.includes("update")) {
        return true;
      }
      return false;
    },
  },

  created() {
    if (localStorage.length !== 0 && !this.isUpdate) {
      localStorage.clear();
    }
    axios.get(`${baseUrl}/api/projects`).then((response) => {
      this.items = response.data;
    });
  },
});
