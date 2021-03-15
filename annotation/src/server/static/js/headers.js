import axios from "axios";
import Vue from "vue";
import Antd from 'ant-design-vue';
import { AutoComplete, Modal, Form, Icon, Button, Radio, Popconfirm, Input, Notification } from "ant-design-vue";
import 'ant-design-vue/dist/antd.css';
import HTTP from "./http";

axios.defaults.xsrfCookieName = "csrftoken";
axios.defaults.xsrfHeaderName = "X-CSRFToken";

Vue.component(Modal.name, Modal)
Vue.use(Modal);
Vue.use(Form);
Vue.use(Icon);
Vue.use(AutoComplete);
Vue.use(Button);
Vue.use(Radio);
Vue.use(Popconfirm);
Vue.use(Input);

Vue.use(Antd);


const vm = new Vue({
  el: '#headers',
  delimiters: ['[[', ']]'],
  data: {
    annotationHeading: null,
    importDataHeading: null,
    modelNamePopupVisible: false,
    modelForm: {
      experiment_name: "",
      params: {},
      "include_documents": true,
    },
    defaultParams: {
      match_batch_size: 50,
      unlabeled_batch_size: 100,
      learning_rate: 0.1,
      epochs: 20,
      embeddings: ["charngram.100d", "fasttext.en.300d", "fasttext.simple.300d", "glove.42B.300d",
        "glove.840B.300d", "glove.twitter.27B.25d", "glove.twitter.27B.50d", "glove.twitter.27B.100d",
        "glove.twitter.27B.200d", "glove.6B.50d", "glove.6B.100d", "glove.6B.200d", "glove.6B.300d"],
      gamma: 0.5,
      hidden_dim: 100,
      random_state: 42,
      load_model: false,
      start_epoch: 0,
      pre_train_hidden_dim: 300,
      pre_train_training_size: 50000,
    }
  },
  methods: {
    showModelTrainingPopup() {
      this.modelNamePopupVisible = true
    },
    trainModel() {
      const embeddingsArr = this.modelForm.params.embeddings.split(".");
      this.modelForm.params['emb_dim'] = parseInt(embeddingsArr[embeddingsArr.length - 1].replace("d", ""));

      HTTP.post('train_model/', this.modelForm).then(res => {
        console.log("train model res", res)
        this.modelNamePopupIsLoading = false
        this.modelNamePopupVisible = false
        this.$notification['success']({
          message: `Model ${this.modelForm.experiment_name} training`,
          description: "Your model is being trained now. You can check the status in the models page"
        })
      }).catch(err => {
        this.$notification['error']({
          message: "Model failed to start training",
          description: "Please try again later"
        })
      })
    }
  },
  created() {
    const projectUrl = window.location.href.split('/').slice(3, 5).join("/");
    axios.get(`/api/${projectUrl}/heading_status`).then((response) => {
      if (response.data.is_annotated) {
        this.annotationHeading = 'Continue Annotating';
      } else {
        this.annotationHeading = 'Annotate Data';
      }
      if (response.data.has_documents) {
        this.importDataHeading = 'Upload More Documents';
      } else {
        this.importDataHeading = 'Upload Documents';
      }
    });
    Object.keys(this.defaultParams).forEach(param => {
      if (Array.isArray(this.defaultParams[param])) {
        this.modelForm.params[param] = this.defaultParams[param][0];
      } else {
        this.modelForm.params[param] = this.defaultParams[param];
      }
    })
  },
});
