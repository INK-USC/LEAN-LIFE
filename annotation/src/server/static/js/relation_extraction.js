/* eslint-disable prefer-destructuring */
import Vue from 'vue';
import VTooltip from 'v-tooltip';
import Loading from 'vue-loading-overlay';
import 'vue-loading-overlay/dist/vue-loading.css';

import annotationMixin from './mixin';
import nlModal from './nlModal';
import triggerModal from './triggerModal';
import relationExtractionAnnotator from './relationExtractionAnnotator';
import relationDisplay from './relationDisplay';
import sentenceHighlightDisplay from './sentenceHighlightDisplay';
import HTTP from './http';
import simpleShortcut from './filter';

Vue.component('nl-modal', nlModal);
Vue.component('trigger-modal', triggerModal);
Vue.component('relation-extraction-annotator', relationExtractionAnnotator);
Vue.component('relation-display', relationDisplay);
Vue.component('sentence-highlight-display', sentenceHighlightDisplay);

// We bind HTTP to the prototype so that the modals can access it, but in an
// ideal world, the modals would not be responsible for making API requests.
Vue.prototype.$http = HTTP;

Vue.use(VTooltip);
Vue.use(Loading);
Vue.use(require('vue-shortkey'), {
  prevent: ['input', 'textarea'],
});

Vue.filter('simpleShortcut', simpleShortcut);
Vue.component('modal', {
  template:
    // eslint-disable-next-line no-multi-str
    '<transition name="modal">\
      <div class="modal-mask"></div>\
    </transition>',
});

const vm = new Vue({
  el: '#mail-app',
  delimiters: ['[[', ']]'],
  components: {
    Loading,
  },
  mixins: [annotationMixin],
  data() {
    return {
      labelsEnabled: false,
      id2LabelDict: {},
      // Vue Errors without this, tries to grab data too early
      reRelations: [],
      subject: null,
      object: null,
      relation: null,
      recommendations: {},
      recommendedLabels: {},
    };
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
          this.annotationDocs[i].extractRelations(this.id2LabelDict);
        }
        // this.allAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
        this.reRelations = this.annotationDocs[this.pageNumber].relationAnnotations;
        this.recommendations = this.annotationDocs[this.pageNumber].getRERecs();
        this.documentsProcessed = false;
      }
    },
    refresh() {
      if (this.refresh) {
        this.reRelations = this.annotationDocs[this.pageNumber].relationAnnotations;
        this.refresh = false;
      }
    },
    pageNumber() {
      if (this.pageNumber < Object.keys(this.annotationDocs).length) {
        this.reRelations = this.annotationDocs[this.pageNumber].relationAnnotations;
      }
    },
    explanationPopup() {
      if (!this.explanationPopup) {
        this.removeRelationHeadings();
      }
    }
  },
  methods: {
    /**
     * Removes a previously-selected relation.
     * @param {Number} i Index to remove at.
     */
    removeSelectedRelation(i) {
      const relationToDelete = this.annotationDocs[this.pageNumber].relationAnnotations[i];
      const docId = this.annotationDocs[this.pageNumber].id;
      HTTP.delete(`docs/${docId}/annotations/${relationToDelete.base_ann_id}`).then(
        () => {
          const update = this.annotationDocs[this.pageNumber].deleteAnnotation(relationToDelete.base_ann_id);
          this.annotationDocs[this.pageNumber].relationAnnotations.splice(i, 1);
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
      );
    },

    /**
     * Clears the subject/object selections in the annotator.
     */
    removeRelationHeadings() {
      this.$refs.annotator.clearClickedChunks();
      this.relation = null;
    },

    disableLabels() {
      this.labelsEnabled = false;
      this.subject = null;
      this.object = null;
      this.recommendedLabels = {};
    },

    enableLabels(chunks) {
      this.subject = chunks[0];
      this.object = chunks[1];
      const key = `${this.subject.start_offset}:${this.subject.end_offset}---${this.object.start_offset}:${this.object.end_offset}`
      if (key in this.recommendations) {
        const recs = this.recommendations[key];
        for (let i=0; i < recs.length; i++) {
          const label = recs[i];
          this.recommendedLabels[label.id] = label;
        }
      }
      this.labelsEnabled = true;
    },

    /**
     * Returns the CSS style to be applied to the label. This is used to display
     * recommended / non-recommended labels differently.
     */
    getLabelStyle(label) {
      let style = {}
      if (!this.labelsEnabled) {
        style =  {
          color: label.text_color,
          backgroundColor: 'grey',
        };
      } else {
        // eslint-disable-next-line no-lonely-if
        if (label.id in this.recommendedLabels) {
          style = {
            color: label.text_color,
            backgroundColor: label.background_color,
            borderStyle: "solid",
            borderColor: "red",
          };
        } else {
          style = {
            color: label.text_color,
            backgroundColor: label.background_color,
          };
        }
      }

      return style;
    },

    checkDuplicate() {
      for (let i=0; i < this.reRelations.length; i++) {
        const relationInfo = this.reRelations[i];
        const subjStartOffSetMatch = relationInfo.subject.start_offset === this.subject.start_offset;
        const subjEndOffSetMatch = relationInfo.subject.end_offset === this.subject.end_offset;
        const objStartOffSetMatch = relationInfo.object.start_offset === this.object.start_offset;
        const objEndOffSetMatch = relationInfo.object.end_offset === this.object.end_offset;
        const labelMatch = relationInfo.relation.id === this.relation.id;
        const total = subjStartOffSetMatch + subjEndOffSetMatch + objStartOffSetMatch + objEndOffSetMatch + labelMatch;
        if (total === 5) {
          return true;
        }
      }

      return false;
    },

    /**
     * Handler function that is called when one of the relation labels is
     * clicked.
     */
    addAnnotation(relation) {
      this.relation = relation;
      if (!this.checkDuplicate()) {
        const sbjFirst = this.subject.start_offset < this.object.start_offset;
        this.selectedWords = (sbjFirst) ? [this.subject.text, this.object.text] : [this.object.text, this.subject.text];
        const baseAnnotation = {
          via_recommendation: false,
          label: relation.id,
        };
        const docId = this.annotationDocs[this.pageNumber].id;
  
        HTTP.post(`docs/${docId}/annotations/`, baseAnnotation).then(
          (response) => {
            this.lastAnnotationId = response.data.id;
            
            const reAnnotation = {
              sbj_start_offset: this.subject.start_offset,
              sbj_end_offset: this.subject.end_offset,
              obj_start_offset: this.object.start_offset,
              obj_end_offset: this.object.end_offset,
            };
  
            this.annotationDocs[this.pageNumber].annotated = true;
  
            // if (!this.onlineLearningIndices.has(docId)) {
            //   this.onlineLearningIndices.add(docId);
            //   this.onlineLearningNum = this.onlineLearningNum + 1;
            // }
  
            HTTP.post(`docs/${docId}/annotations/${this.lastAnnotationId}/re/`, reAnnotation).then(
              (reResponse) => {
                const relationData = {
                  type: "re",
                  base_ann_id: this.lastAnnotationId,
                  sbj_start_offset: reResponse.data.sbj_start_offset,
                  sbj_end_offset: reResponse.data.sbj_end_offset,
                  obj_start_offset: reResponse.data.obj_start_offset,
                  obj_end_offset: reResponse.data.obj_end_offset,
                  label: relation.id,
                };
                this.annotationDocs[this.pageNumber].annotations[relationData.base_ann_id] = relationData;
                this.annotationDocs[this.pageNumber].relationAnnotations.push({
                  subject: this.subject, object: this.object, relation, base_ann_id: this.lastAnnotationId,
                });
                if (this.isNewlyCreated === 0) {
                  this.explanationPopup = true;
                } else if (this.explanationType > 1) {
                  this.isNewlyCreated += 1;
                }
                this.refresh = true;
              },
            ).then(
              () => {
                if (this.annotationDocs[this.pageNumber].relationAnnotations.length === 1) {
                  HTTP.patch(`docs/${docId}`, { annotated: true });
                }
              },
            ).then(
              () => {
                const history = {
                  word_1: this.subject.text,
                  word_2: this.object.text,
                  label: relation.id,
                  annotation: this.lastAnnotationId,
                  user_provided: false,
                };
                HTTP.post("history/re/", history);
                if (this.explanationType === 1) {
                  this.removeRelationHeadings();
                }
              },
            )
              .catch(
                (err) => {
                  console.log(err);
                  HTTP.delete(`docs/${docId}/annotations/${this.lastAnnotationId}`);
                },
              );
          },
        );
      } else {
        console.log("You have already added this relation");
        this.relation = null;
      }
    },

    toggleExplanations(subject, object, relation, annotationId) {
      if (this.explanationType > 0) {
        this.subject = subject;
        this.object = object;
        this.relation = relation;
        const sbjFirst = subject.start_offset < object.start_offset;
        this.selectedWords = (sbjFirst) ? [this.subject.text, this.object.text] : [this.object.text, this.subject.text];
        this.explanationPopup = true;
        this.lastAnnotationId = annotationId;
      }
    },
  },
});
