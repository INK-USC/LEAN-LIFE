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
        modelName: "",
        settings: {
            key1: "10",
            key2: "20",
            key3: "30"
        }
    }
  },
  methods: {
    showModelTrainingPopup(){
        this.modelNamePopupVisible=true
    },
    trainModel(){
        HTTP.post('train_model/',this.modelForm).then(res=>{
          console.log("train model res", res)
          this.modelNamePopupIsLoading=false
          this.modelNamePopupVisible=false
          this.$notification['success']({
            message: `Model ${this.modelForm.modelName} training`,
            description: "Your model is being trained now. You can check the status in the models page"
          })
        }).catch(err=>{
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
  },
});
