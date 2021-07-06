import Vue from 'vue';
import VTooltip from 'v-tooltip';
import Loading from 'vue-loading-overlay';
import 'vue-loading-overlay/dist/vue-loading.css';

import annotationMixin from './mixin';
import nlModal from './nlModal';
import triggerModal from './triggerModal';
import sentenceLabelDisplay from './sentenceLabelDisplay';
import HTTP from './http';
import simpleShortcut from './filter';

Vue.component('nl-modal', nlModal);
Vue.component('trigger-modal', triggerModal);
Vue.component('sentence-label-display', sentenceLabelDisplay);

// We bind HTTP to the prototype so that the modals can access it, but in an
// ideal world, the modals would not be responsible for making API requests.
Vue.prototype.$http = HTTP;

Vue.use(VTooltip);
Vue.use(Loading);
Vue.use(require('vue-shortkey'), {
  prevent: ['input', 'textarea'],
});

Vue.filter('simpleShortcut', simpleShortcut);

const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  components: {
    Loading,
  },
  mixins: [annotationMixin],
  data: {
    // The most-recently selected label by the user. Used in determining which
    // label to show in the modal and which label to attach to explanations.
    currSelectedLabel: null,
    // Recommended labels. TODO: May have to move this into a computed prop or
    // method later.
    // recommendedLabels: [],
    // Labels currently selected by the user. List of label objects.
    // selectedLabels: [],
    id2LabelDict: {},
    saAnnotations: [],
  },
  watch: {
    documentsProcessed() {
      if (this.documentsProcessed) {
        for (let i = 0; i < this.labels.length; i++) {
          const label = this.labels[i];
          this.id2LabelDict[label.id] = label;
        }
        const pageNumbers = Object.keys(this.annotationDocs);
        for (let i = 0; i < pageNumbers.length; i++) {
          this.annotationDocs[i].addLabelInfoToAnnotation(this.id2LabelDict);
        }
        this.saAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
        this.documentsProcessed = false;
      }
    },
    refresh() {
      if (this.refresh) {
        this.saAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
        this.refresh = false;
      }
    },
    pageNumber() {
      if (this.pageNumber < Object.keys(this.annotationDocs).length) {
        this.saAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
      }
    },
  },
  methods: {
    /**
     * Applies the current label as an annotation to the document.
     */
    addAnnotation(label) {
      if (this.isSelected(label)) {
        return;
      }
      this.currSelectedLabel = label;
      const baseAnnotation = {
        via_recommendation: false,
        label: label.id,
      }
      const docId = this.annotationDocs[this.pageNumber].id;
      HTTP.post(`docs/${docId}/annotations/`, baseAnnotation).then((response) => {
        // Annotation ID will come from the response.
        this.lastAnnotationId = response.data.id;
        this.annotationDocs[this.pageNumber].annotated = true;

        HTTP.post(`docs/${docId}/annotations/${this.lastAnnotationId}/sa/`, {})
          .then(() => {
            this.annotationDocs[this.pageNumber].annotations[this.lastAnnotationId] = { label, base_ann_id: this.lastAnnotationId };
            if (this.isNewlyCreated === 0) {
              this.explanationPopup = true;
            } else if (this.explanationType > 1) {
              this.isNewlyCreated += 1;
            }
            this.refresh = true;
          })
          .then(() => {
            if (Object.keys(this.annotationDocs[this.pageNumber].annotations).length === 1) {
              HTTP.patch(`docs/${docId}`, { annotated: true }).then(()=>{
                this.getDocuments();
              });
            }
          })
          .catch((err) => {
            console.log(err);
            HTTP.delete(`docs/${docId}/annotations/${this.lastAnnotationId}`);
          });
      });
    },

    /**
     * Removes the given label from the document.
     */
    removeAnnotation(annotation) {
      const docId = this.annotationDocs[this.pageNumber].id;
      HTTP.delete(`docs/${docId}/annotations/${annotation.base_ann_id}`).then(
        (response) => {
          const update = this.annotationDocs[this.pageNumber].deleteAnnotation(annotation.base_ann_id);
          this.refresh = true;
          if (update) {
            // if (this.onlineLearningIndices.has(docId)) {
            //   this.onlineLearningIndices.delete(docId);
            //   this.onlineLearningNum = this.onlineLearningNum - 1;
            // }
            HTTP.patch(`docs/${docId}`, { annotated: false }).then(
              () => {},
            );
          }
        },
      ).then(()=>{
        this.getDocuments()
      });
    },

    /**
     * Returns true if the given label is recommended.
     */
    isRecommended(label) {
      // TODO: Change this placeholder.
      return false;
    },

    /**
     * Returns true if the given label has already been selected.
     */
    isSelected(label) {
      return this.saAnnotations.map(l => l.label.id).indexOf(label.id) > -1;
    },

    /**
     * Returns the CSS style to be applied to the label. This is used to display
     * recommended / non-recommended labels differently.
     */
    getLabelStyle(label) {
      return {
        color: label.text_color,
        backgroundColor: this.isSelected(label) ? 'grey' : label.background_color,
      };
    },

    toggleExplanation(annotation) {
      this.lastAnnotationId = annotation.base_ann_id;
      this.currSelectedLabel = annotation.label;
      this.explanationPopup = true;
    },
  },
  computed: {
    /**
     * Returns the labels, sorted by ascending label ID. Side effect is that it
     * also modifies this.labels since sort operates in-place, but that should
     * be fine.
     */
    sortedLabels() {
      return this.labels.sort((l1, l2) => l1.id - l2.id);
    },
  },
});
