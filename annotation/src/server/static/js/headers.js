import axios from "axios";
import Vue from "vue";

axios.defaults.xsrfCookieName = "csrftoken";
axios.defaults.xsrfHeaderName = "X-CSRFToken";

const vm = new Vue({
  el: '#headers',
  delimiters: ['[[', ']]'],
  data: {
    annotationHeading: null,
    importDataHeading: null,
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
