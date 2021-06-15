import Vue from 'vue';
import HTTP from './http';
import simpleShortcut from './filter';

Vue.filter('simpleShortcut', simpleShortcut);


const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  data: {
    labels: [],
    labelText: '',
    selectedKey: '',
    checkedKey: [],
    shortcutKey: '',
    backgroundColor: '#209cee',
    textColor: '#ffffff',
    errorMessage: "",
    labelCreationStarted: false,
    newlyCreated: false,
  },

  watch: {
    labelText() {
      if (this.labelText.length > 0) {
        this.labelCreationStarted = true;
      } else {
        this.labelCreationStarted = false;
      }
    },
    shortcutKey() {
      if (this.selectedKey.length !== '') {
        this.labelCreationStarted = true;
      } else {
        this.labelCreationStarted = false;
      }
    },
    checkedKey() {
      if (this.checkedKey.length > 0) {
        this.labelCreationStarted = true;
      } else {
        this.labelCreationStarted = false;
      }
    },
    backgroundColor() {
      if (this.backgroundColor !== '#209cee') {
        this.labelCreationStarted = true;
      } else {
        this.labelCreationStarted = false;
      }
    },
    textColor() {
      if (this.textColor !== '#ffffff') {
        this.labelCreationStarted = true;
      } else {
        this.labelCreationStarted = false;
      }
    },
  },


  computed: {
    /**
      * combineKeys: Combine selectedKey and checkedKey to get shortcutKey
      * saveKeys: Save null to database if shortcutKey is empty string
      */
    combineKeys() {
      this.shortcutKey = '';

      // If checkedKey exits, add it to shortcutKey
      if (this.checkedKey.length > 0) {
        this.checkedKey.sort();
        this.shortcutKey = this.checkedKey.join(' ');

        // If selectedKey exist, add it to shortcutKey
        if (this.selectedKey.length !== 0) {
          this.shortcutKey = this.shortcutKey + ' ' + this.selectedKey;
        }
      }

      // If only selectedKey exist, assign to shortcutKey
      if (this.shortcutKey.length === 0 && this.selectedKey.length !== 0) {
        this.shortcutKey = this.selectedKey;
      }
      return this.shortcutKey;
    },

    saveKeys() {
      this.shortcutKey = this.combineKeys;
      if (this.shortcutKey === '') {
        return null;
      }
      return this.shortcutKey;
    },
  },

  methods: {
    addLabel() {
      const payload = {
        text: this.labelText,
        shortcut: this.saveKeys,
        background_color: this.backgroundColor,
        text_color: this.textColor,
      };
      if (this.labels.length == 0 && JSON.parse(window.localStorage.getItem("newly_created")) === true) {
        document.getElementById("left_nav").style.pointerEvents = "auto";
        document.getElementById("top_nav").style.pointerEvents = "auto";
      }
      HTTP.post('labels/', payload).then((response) => {
        this.reset();
        this.labels.push(response.data);
        this.errorMessage = "";
      }).catch((error) => {
        if (error.response.status == 500) {
          const errorType = error.response.request.response.split(':', 2)[1].split('(')[1].split(')')[0].split(",")[1].substring(1);
          console.log(errorType);
          if (errorType == "text") {
            this.errorMessage = "The label already exists!";
          } else if (errorType == "shortcut") {
            this.errorMessage = "The Shortcut key already exists!";
          }
        }
      });
    },

    removeLabel(label) {
      const labelId = label.id; 
      HTTP.delete(`labels/${labelId}`).then((response) => {
        const index = this.labels.indexOf(label);
        this.labels.splice(index, 1);
      });
    },

    reset() {
      this.labelText = '';
      this.selectedKey = '';
      this.checkedKey = [];
      this.shortcutKey = '';
      this.backgroundColor = '#209cee';
      this.textColor = '#ffffff';
      this.errorMessage = "";
    },
    done() {
      const projectUrl = window.location.href.split('/').slice(0, 5).join("/") + "/setting/";
      window.location.replace(projectUrl);
    },
  },
  created() {
    HTTP.get('labels').then((response) => {
      this.labels = response.data;
    });
    if (JSON.parse(window.localStorage.getItem("newly_created")) === true) {
      document.getElementById("left_nav").style.pointerEvents = "none";
      document.getElementById("top_nav").style.pointerEvents = "none";
      this.newlyCreated = true;
    }
  },
});
