/* eslint-disable key-spacing */
import Vue from "vue";
import VTooltip from "v-tooltip";
import Loading from "vue-loading-overlay";
import "vue-loading-overlay/dist/vue-loading.css";

import annotationMixin from "./mixin";
import nlModal from "./nlModal";
import triggerModal from "./triggerModal";
import sentenceHighlightDisplay from "./sentenceHighlightDisplay";
import HTTP from "./http";
import simpleShortcut from "./filter";

Vue.component("nl-modal", nlModal);
Vue.component("trigger-modal", triggerModal);
Vue.component("sentence-highlight-display", sentenceHighlightDisplay);

// We bind HTTP to the prototype so that the modals can access it, but in an
// ideal world, the modals would not be responsible for making API requests.
Vue.prototype.$http = HTTP;

Vue.use(VTooltip);
Vue.use(Loading);
Vue.use(require("vue-shortkey"), {
  prevent: ["input", "textarea"],
});

Vue.filter("simpleShortcut", simpleShortcut);

Vue.component("annotator", {
  template:
    // eslint-disable-next-line no-multi-str
    '<div v-on:mouseup="setSelectedRange"> \
      <span class="text-sequence" v-for="chunk in chunks" \
        v-bind:class="{tag: id2label[chunk.label].text_color}"  \
        v-bind:style="{ color: id2label[chunk.label].text_color, backgroundColor: id2label[chunk.label].background_color, textDecoration: id2label[chunk.label].text_decoration}">\
          <span v-if="id2label[chunk.label].text_color" v-on:click="inspectExplanations(chunk)">{{ text.slice(chunk.start_offset, chunk.end_offset) }}</span>\
          <span v-else>{{ text.slice(chunk.start_offset, chunk.end_offset) }}</span> \
          <button class="delete is-small" v-if="id2label[chunk.label].text_color" @click="removeAnnotation(chunk)"></button> \
      </span> \
    </div>',
  props: {
    labels: Array, // [{id: Integer, color: String, text: String}]
    text: String,
    entityPositions: Array, // [{'start_offset': 10, 'end_offset': 15, 'label': 1}]
  },

  methods: {
    setSelectedRange() {
      let selectionStart = null;
      let selectionEnd = null;

      if (window.getSelection().anchorOffset !== window.getSelection().focusOffset) {
        const range = window.getSelection().getRangeAt(0);
        selectionStart = range.startOffset;
        selectionEnd = range.endOffset;

        const leadingWhiteSpace = range.commonAncestorContainer.data.search(/\S/);
        selectionStart -= leadingWhiteSpace;
        selectionEnd -= leadingWhiteSpace;

        const spanText = range.commonAncestorContainer.data.trim();
        const offSet = this.text.search(spanText);

        selectionStart += offSet;
        selectionEnd += offSet;
      } else {
        this.resetSelection();
      }

      // A selection has been made
      if (selectionStart !== null && selectionEnd !== null) {
        // Trimming if Needed
        while (true) {
          if (!/[a-zA-Z0-9]/.test(this.text.slice(selectionStart, selectionStart+1))) {
            selectionStart += 1;
          } else if (!/[a-zA-Z0-9]/.test(this.text.slice(selectionEnd-1, selectionEnd))) {
            selectionEnd -= 1;
          } else {
            break;
          }
        }

        // Start of word not included
        while (selectionStart > 0 && /[a-zA-Z0-9-_]/.test(this.text.slice(selectionStart-1, selectionStart))) {
          selectionStart -= 1;
        }

        // End of word not included
        while (selectionEnd < this.text.length && /[a-zA-Z0-9-]/.test(this.text.slice(selectionEnd, selectionEnd+1))) {
          selectionEnd += 1;
        }
        this.$emit("set-selection", selectionStart, selectionEnd);
      }

    },

    resetSelection() {
      this.$emit("set-selection", 0, 0);
    },

    removeAnnotation(chunk) {
      this.$emit("remove-annotation", chunk.base_ann_id);
    },

    makeLabel(startOffset, endOffset) {
      const label = {
        id: 0,
        label: -1,
        start_offset: startOffset,
        end_offset: endOffset,
      };
      return label;
    },

    inspectExplanations(chunk) {
      this.$emit("toggle-explanations", chunk);
    },
  },

  computed: {
    sortedEntityPositions() {
      this.entityPositions = this.entityPositions.sort(
        (a, b) => a.start_offset - b.start_offset
      );
      return this.entityPositions;
    },

    chunks() {
      const res = [];
      let left = 0;
      for (let i = 0; i < this.sortedEntityPositions.length; i++) {
        const e = this.sortedEntityPositions[i];
        const l = this.makeLabel(left, e.start_offset);
        res.push(l);
        res.push(e);
        left = e.end_offset;
      }
      const l = this.makeLabel(left, this.text.length);
      res.push(l);

      return res;
    },

    id2label() {
      const id2label = {};
      // default value;
      id2label[-1] = {
        text_color: "",
        background_color: "",
        text_decoration: "",
      };

      for (let i = 0; i < this.labels.length; i++) {
        const label = this.labels[i];
        label.text_decoration = "";
        id2label[label.id] = label;
      }
      return id2label;
    },
  },
});

Vue.component("recommender", {
  template: `
  <div>
    <template v-for="r in recommendationChunks">
      <v-popover style="display: inline" offset="4" v-if="r.text_decoration">
        <span class="text-sequence tooltip-target" 
              v-bind:class="{tag: r.label ? 'r.label.text_color' : ''}"
              v-bind:style="{textDecoration: r.text_decoration}">
              {{ text.slice(r.start_offset, r.end_offset) }}
        </span>
        <template slot="popover">
          <div class="card-header-title has-background-royalblue" style="padding:1.5rem;">
            <div class="field is-grouped is-grouped-multiline">
              <div class="control" v-for="label in labels" v-bind:style="[r.label ? r.label.text==label.text ? {'border': 'solid red'} : {'border': 'none'} : {'border': 'none'}]">
                <div class="tags has-addons">
                  <a class="tag is-medium" v-bind:style="{ color: label.text_color, backgroundColor: label.background_color }"
                                          v-on:click="addAnnotation(label.id, r.start_offset, r.end_offset)">{{ label.text }}</a>
                  <span class="tag is-medium"><kbd>{{ label.shortcut | simpleShortcut }}</kbd></span>
                </div>
              </div>
            </div>
          </div>
        </template>
      </v-popover>
      <span v-else class="text-sequence"
          v-bind:class="{tag: r.label ? r.label.text_color : ''}"
          v-bind:style="{textDecoration: r.text_decoration}">{{ text.slice(r.start_offset, r.end_offset) }}</span>
      </span>
    </template>
  </div>`,

  props: {
    labels: Array, // [{id: Integer, color: String, text: String}]
    text: String,
    recommendPositions: Array,
  },

  methods: {
    addAnnotation(labelId, startOffset, endOffset) {
      this.$emit("add-annotation", labelId, startOffset, endOffset, true);
    },
  },

  computed: {
    sortedRecommendPositions() {
      this.recommendPositions.sort((a, b) => a.start_offset - b.start_offset);
      return this.recommendPositions;
    },

    recommendationChunks() {
      const chunks = [];
      if (this.sortedRecommendPositions.length) {
        for (let i = 0; i < this.sortedRecommendPositions.length; i++) {
          const e = this.sortedRecommendPositions[i];
          e.text_decoration = "underline";
          if (i === 0) {
            if (e.start_offset !== 0) {
              chunks.push({
                text_decoration: "",
                start_offset: 0,
                end_offset: e.start_offset,
              });
            }
          } else {
            chunks.push({
              text_decoration: "",
              start_offset: this.sortedRecommendPositions[i - 1].end_offset,
              end_offset: e.start_offset,
            });
          }
          chunks.push(e);
        }
        const lastRec = this.sortedRecommendPositions[
          this.sortedRecommendPositions.length - 1
        ];
        chunks.push({
          text_decoration: "",
          start_offset: lastRec.end_offset,
          end_offset: this.text.length,
        });
      }
      return chunks;
    },
  },
});

const vm = new Vue({
  el: "#mail-app",
  delimiters: ["[[", "]]"],
  components: {
    Loading,
  },
  mixins: [annotationMixin],
  
  data: {
    nerAnnotations: [],
    startOffset: 0,
    endOffset: 0,
  },

  watch: {
    refresh() {
      if (this.refresh) {
        this.nerAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
        this.startOffset = 0;
        this.endOffset = 0;
        this.refresh = false;
      }
    },
    pageNumber() {
      if (this.pageNumber < Object.keys(this.annotationDocs).length) {
        this.nerAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
      }
    },
    documentsProcessed() {
      if (this.documentsProcessed) {
        this.nerAnnotations = this.annotationDocs[this.pageNumber].getAnnotationList();
        this.documentsProcessed = false;
      }
    },
  },

  methods: {
    duplicateRange(startOffset, endOffset) {
      for (let i = 0; i < this.nerAnnotations.length; i++) {
        const e = this.nerAnnotations[i];
        if (e.start_offset === startOffset && e.end_offset === endOffset) {
          return e.base_ann_id;
        }
      }
      return false;
    },

    validRange(startOffset, endOffset) {
      // eslint-disable-next-line prefer-destructuring
      const text = this.annotationDocs[this.pageNumber].text;
      
      if (startOffset === endOffset) {
        return false;
      }
      if (startOffset > text.length || endOffset > text.length) {
        return false;
      }
      if (startOffset < 0 || endOffset < 0) {
        return false;
      }
      for (let i = 0; i < this.nerAnnotations.length; i++) {
        const e = this.nerAnnotations[i];
        if (e.start_offset <= startOffset && startOffset < e.end_offset) {
          return false;
        }
        if (e.start_offset < endOffset && endOffset < e.end_offset) {
          return false;
        }
        if (startOffset < e.start_offset && e.start_offset < endOffset) {
          return false;
        }
        if (startOffset < e.end_offset && e.end_offset < endOffset) {
          return false;
        }
      }
      return true;
    },

    addAnnotation(labelId, startOffset, endOffset, recommendation) {
      const duplicateResponse = this.duplicateRange(startOffset, endOffset);
      if (duplicateResponse !== false) {
        this.removeAnnotation(duplicateResponse);
      }

      if (this.validRange(startOffset, endOffset)) {
        this.selectedWords = [this.annotationDocs[this.pageNumber].text.slice(
          startOffset,
          endOffset,
        )];

        const baseAnnotation = {
          via_recommendation: recommendation,
          label: labelId,
        };

        const docId = this.annotationDocs[this.pageNumber].id;
        HTTP.post(`docs/${docId}/annotations/`, baseAnnotation).then(
          (response) => {
            this.lastAnnotationId = response.data.id;
            const nerAnnotation = {
              start_offset: startOffset,
              end_offset: endOffset,
              user_provided: false,
            };

            this.annotationDocs[this.pageNumber].annotated = true;

            // if (!this.onlineLearningIndices.has(docId)) {
            //   this.onlineLearningIndices.add(docId);
            //   this.onlineLearningNum = this.onlineLearningNum + 1;
            // }

            HTTP.post(
              `docs/${docId}/annotations/${this.lastAnnotationId}/ner/`,
              nerAnnotation
            )
              .then((nerResponse) => {
                const mentionData = {
                  base_ann_id: this.lastAnnotationId,
                  start_offset: nerResponse.data.start_offset,
                  end_offset: nerResponse.data.end_offset,
                  label: labelId,
                };
                this.annotationDocs[this.pageNumber].annotations[mentionData.base_ann_id] = mentionData;
                if (this.isNewlyCreated === 0) {
                  this.explanationPopup = true;
                } else if (this.explanationType > 1) {
                  this.isNewlyCreated += 1;
                }
                this.refresh = true;
              })
              .then(() => {
                if (Object.keys(this.annotationDocs[this.pageNumber].annotations).length === 1) {
                  HTTP.patch(`docs/${docId}`, { annotated: true });
                }
              })
              .then(() => {
                const history = {
                  word: this.$refs.annotator.text.slice(startOffset, endOffset),
                  label: labelId,
                  annotation: this.lastAnnotationId,
                  user_provided: false,
                };
                HTTP.post("history/ner/", history);
              })
              .catch((err) => {
                console.log(err);
                HTTP.delete(`docs/${docId}/annotations/${this.lastAnnotationId}`);
              });
          },
        );
      }
    },

    removeAnnotation(annotationId) {
      const docId = this.annotationDocs[this.pageNumber].id;
      HTTP.delete(`docs/${docId}/annotations/${annotationId}`).then(
        (response) => {
          const update = this.annotationDocs[this.pageNumber].deleteAnnotation(annotationId);
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

    toggleExplanations(chunk) {
      if (this.explanationType > 1) {
        this.selectedWords = [this.annotationDocs[this.pageNumber].text.slice(
          chunk.startOffset,
          chunk.chunkendOffset,
        )];
        this.lastAnnotationId = chunk.base_ann_id;
        this.explanationPopup = true;
      }
    },

    setSelection(startOffset, endOffset) {
      this.startOffset = startOffset;
      this.endOffset = endOffset;
    },
  },
});
