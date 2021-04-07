<template>
  <el-card style="text-align: left; margin-top: 20px">
    <el-row>
      <el-tag>Text</el-tag>
    </el-row>

    <el-row style="line-height: 4">
      <div v-on:mouseup="setSelectedRange">
        <span v-for="(chunk, index) in this.chunks" :key="index"
              :style="{color: id2label[chunk.label] ? id2label[chunk.label].text_color: '',
              backgroundColor: id2label[chunk.label] ? id2label[chunk.label].background_color : '',
              padding: '10px 10px 10px 10px', fontSize: '18px',
              borderRadius: id2label[chunk.label]? (id2label[chunk.label].text_color? '8px': 0) : 0}"
        >
          <span @click="onAnnotationClicked(chunk)">
            {{ fullText.slice(chunk.start_offset, chunk.end_offset) }}
          </span>
          <i v-if="id2label[chunk.label] && id2label[chunk.label].text_color" class="el-icon-delete"
             style="cursor: pointer" @click="removeAnnotation(chunk)"/>
        </span>
      </div>
    </el-row>

    <Recommendation/>
  </el-card>
</template>

<script>
import Recommendation from "@/components/project/annotation/shared/Recommendation";
import {ACTION_TYPE, DIALOG_TYPE} from "@/utilities/constant";
// annotation for NER
export default {
  name: "NamedEntityRecognitionAnnotation",
  components: {Recommendation},
  data() {
    return {}
  },
  methods: {
    setSelectedRange() {
      let selectionStart = null;
      let selectionEnd = null;
      console.log("selection", window.getSelection(), document.getSelection())

      if (window.getSelection().anchorOffset !== window.getSelection().focusOffset) {
        const range = window.getSelection().getRangeAt(0);
        console.log("range", range);

        selectionStart = range.startOffset;
        selectionEnd = range.endOffset;

        const leadingWhiteSpace = range.commonAncestorContainer.data.search(/\S/);
        selectionStart -= leadingWhiteSpace;
        selectionEnd -= leadingWhiteSpace;

        const spanText = range.commonAncestorContainer.data.trim();
        const offSet = this.$store.getters["document/getCurDoc"].text.search(spanText);
        console.log("this.txt", this.$store.getters["document/getCurDoc"].text)
        console.log("span text", spanText);
        console.log("offset", offSet)

        selectionStart += offSet;
        selectionEnd += offSet;
      } else {
        console.log("reset selection");
        selectionStart = null;
        selectionEnd = null;
        this.$store.dispatch('annotation/resetNERSelection')
      }

      // A selection has been made
      if (!!selectionStart && !!selectionEnd) {
        // Trimming if Needed
        while (true) {
          if (!/[a-zA-Z0-9]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionStart, selectionStart + 1))) {
            selectionStart += 1;
          } else if (!/[a-zA-Z0-9]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionEnd - 1, selectionEnd))) {
            selectionEnd -= 1;
          } else {
            break;
          }
        }
        // Start of word not included
        while (selectionStart > 0 && /[a-zA-Z0-9-_]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionStart - 1, selectionStart))) {
          selectionStart -= 1;
        }
        // End of word not included
        while (selectionEnd < this.$store.getters["document/getCurDoc"].text.length && /[a-zA-Z0-9-]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionEnd, selectionEnd + 1))) {
          selectionEnd += 1;
        }
        console.log("selection start end", selectionStart, selectionEnd)
        this.$store.dispatch('annotation/setNERSelection', {
          "selectionStart": selectionStart,
          "selectionEnd": selectionEnd
        })
      }
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
    removeAnnotation(annotation) {
      console.log("remove annotation", annotation)
      this.$http
          .delete(`projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${annotation.base_ann_id}`)
          .then(res => {
            if (this.$store.getters["document/getCurDoc"].annotations.length == 1) {
              return this.$http
                  .patch(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}`,
                      {annotated: false}
                  )
            }
          })
          .then(() => {
            this.$store.dispatch('document/fetchDocuments')
          })
    },
    onAnnotationClicked(chunk) {
      console.log("chunk clicked", chunk, this.id2label[chunk.label])
      if (chunk.label === -1) {
        return;
      }
      this.$store.dispatch("annotation/setNERSelection", {
        selectionStart: chunk.start_offset,
        selectionEnd: chunk.end_offset
      }).then(() => {
        this.$store.dispatch("explanation/showExplanationPopup", {annotationId: chunk.base_ann_id})
      });
    }
  },
  created() {
    // console.log(this.$store.getters["document/getCurDoc"])
    if (this.$store.getters.getActionType === ACTION_TYPE.CREATE) {
      this.$store.commit("showAnnotationGuidePopup", DIALOG_TYPE.Annotation.NER);
    }
  },
  computed: {
    chunks() {
      if (!this.$store.getters["document/getCurDoc"]) {
        return [];
      }
      let sortedEntityPositions = this.$store.getters["document/getCurDoc"] ? this.$store.getters["document/getCurDoc"].formattedAnnotations : [];
      const res = [];
      let left = 0;
      for (let i = 0; i < sortedEntityPositions.length; i++) {
        const e = sortedEntityPositions[i];
        const l = this.makeLabel(left, e.start_offset);
        res.push(l);
        res.push(e);
        left = e.end_offset;
      }
      const l = this.makeLabel(left, this.$store.getters["document/getCurDoc"].text.length);
      res.push(l);
      return res
    },
    fullText() {
      return this.$store.getters["document/getCurDoc"].text;
    },
    id2label() {
      const id2label = {};
      // default value;
      id2label[-1] = {
        text_color: "",
        background_color: "",
        text_decoration: "",
      };

      for (let i = 0; i < this.$store.getters['label/getLabels'].length; i++) {
        const label = this.$store.getters["label/getLabels"][i];
        label.text_decoration = "";
        id2label[label.id] = label;
      }
      return id2label;
    },


  }
}
</script>

<style scoped>

</style>
