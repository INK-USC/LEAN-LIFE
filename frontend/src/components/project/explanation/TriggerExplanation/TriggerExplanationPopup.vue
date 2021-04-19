<template>
  <el-dialog :visible.sync="dialogVisible" v-if="!!this.$store.getters['document/getCurDoc'] "
             class="explanation-dialog" @open="dialogOpen" @close="dialogClose">
    <h1 slot="title">Trigger Explanation</h1>

    <el-divider/>

    <div style="text-align: initial">
      <SentimentAnalysisBrief v-if="this.$store.getters.getProjectInfo.task===1"/>
      <NamedEntityRecognitionBrief v-if="this.$store.getters.getProjectInfo.task===2"/>
      <RelationExtractionBrief v-if="this.$store.getters.getProjectInfo.task===3"/>

      <el-form style="margin-top: 20px">
        <el-form-item v-for="(trigger, triggerIndex) in this.triggers" :key="triggerIndex">

          <div style="display: flex; flex-display:row; justify-content: space-between; align-items: center">
            <h4>Trigger {{ triggerIndex + 1 }}</h4>
            <i class="el-icon-delete" style="cursor: pointer" @click="removeTrigger(triggerIndex)"/>
          </div>
          <el-row style="line-height: 2">
            <div @mouseup="setSelectedRange(triggerIndex)">
              <span v-for="(chunk, chunkIndex) in chunks" :key="chunkIndex" :style=getChunkStyle(chunk,triggerIndex)>
                <span>
                  {{ $store.getters["document/getCurDoc"].text.slice(chunk.start_offset, chunk.end_offset) }}
                </span>
                <i class="el-icon-delete" style="cursor:pointer;"
                   @click="removeReason(triggerIndex, chunkIndex, chunk)"
                   v-if="chunk.labelId === triggerIndex"/>
              </span>
            </div>
          </el-row>

        </el-form-item>
        <el-form-item>
          <el-button @click="addAnotherExplanation" style="width: 100%; border: 1px dotted" type="primary"
                     :disabled="!canAddExplanation">
            <i class="el-icon-plus"/> Add additional explanation
          </el-button>
        </el-form-item>
      </el-form>

    </div>

    <span slot="footer">
      <el-button @click="$store.dispatch('explanation/hideExplanationPopup')">Cancel</el-button>
      <el-button type="primary" @click="submitExplanations">OK</el-button>
    </span>
  </el-dialog>
</template>

<script>
import RelationExtractionBrief from "@/components/project/explanation/brief/re/RelationExtractionBrief";
import NamedEntityRecognitionBrief from "@/components/project/explanation/brief/ner/NamedEntityRecognitionBrief";
import SentimentAnalysisBrief from "@/components/project/explanation/brief/sa/SentimentAnalysisBrief";
// Const def to represent an unlabeled chunk.
const UNLABELED_LABEL_ID = -1;
// Const def to represent a chunk that is the original selected text.
const SELECTION_LABEL_ID = -2;
// Const def to represent a chunk that is the original selected text.
const NOT_YET_PUSHED_ID = -3;

// popup for trigger explanation. will be triggered when user added new annotation and the explanation type is trigger explanation.
export default {
  name: "TriggerExplanationPopup",
  components: {RelationExtractionBrief, NamedEntityRecognitionBrief, SentimentAnalysisBrief},
  created() {
    this.generateTriggerLabels();
  },
  data() {
    return {
      triggers: [],
      reasonsToDelete: [],
      triggerLabels: [],
    }
  },
  methods: {
    dialogOpen() {
      if (this.$store.getters["explanation/getAnnotationInfo"]) {
        this.triggers = this.$store.getters["explanation/getAnnotationInfo"].formattedTriggerExplanation;
        console.log("reasons", this.triggers)
      }
      if (!this.triggers || this.triggers.length === 0 || !this.triggers[0] || this.triggers[0].length == 0) {
        this.triggers = [[{
          start_offset: -1,
          end_offset: -1,
          pk_id: NOT_YET_PUSHED_ID,
          labelId: -1,
          base_ann_id: this.$store.getters["explanation/getAnnotationInfo"].extended_annotation.annotation_id,
        }]]
      }
    },
    dialogClose() {
      console.log("dialog close")
      this.triggers = [[]];
      this.reasonsToDelete = [];
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
    // add another explanation
    addAnotherExplanation() {
      this.triggers.push([{
        start_offset: -1,
        end_offset: -1,
        pk_id: NOT_YET_PUSHED_ID,
        labelId: -1,
        base_ann_id: this.$store.getters["explanation/getAnnotationInfo"].extended_annotation.annotation_id,
      }]);
      console.log("added", this.triggers)
    },
    // update selected the range when user highlight words
    setSelectedRange(triggerIndex) {
      let selectionStart = null;
      let selectionEnd = null;
      // console.log("selection", window.getSelection(), document.getSelection())
      if (window.getSelection().anchorOffset !== window.getSelection().focusOffset) {
        const range = window.getSelection().getRangeAt(0);
        // console.log("range", range);

        selectionStart = range.startOffset;
        selectionEnd = range.endOffset;

        const leadingWhiteSpace = range.commonAncestorContainer.data.search(/\S/);
        selectionStart -= leadingWhiteSpace;
        selectionEnd -= leadingWhiteSpace;

        const spanText = range.commonAncestorContainer.data.trim();
        const offSet = this.$store.getters["document/getCurDoc"].text.search(spanText);
        // console.log("this.txt", this.$store.getters["document/getCurDoc"].text)
        // console.log("span text", spanText);
        // console.log("offset", offSet)

        selectionStart += offSet;
        selectionEnd += offSet;
      } else {
        console.log("reset selection")
      }

      // A selection has been made
      if (selectionStart !== null && selectionEnd != null) {
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
        console.log("selection text", this.$store.getters["document/getCurDoc"].text.slice(selectionStart, selectionEnd))
        // console.log("reasons", this.triggers)
        const curTrigger = this.triggers[triggerIndex]
        console.log("curtrigger", curTrigger)
        let curExplanation = curTrigger[curTrigger.length - 1]
        console.log("cur exp", curExplanation)
        if (curExplanation.start_offset === -1 || curExplanation.end_offset === -1) {
          // console.log("already exist empty offset")
          curExplanation.start_offset = selectionStart;
          curExplanation.end_offset = selectionEnd;
          curExplanation.labelId = triggerIndex;
        } else {
          // check if it is a subselection
          const canSelect = !curTrigger.some(trig => (trig.start_offset <= selectionStart && selectionEnd <= trig.end_offset))
          if (canSelect){
            console.log("adding new offset")
            curTrigger.push({
              start_offset: selectionStart,
              end_offset: selectionEnd,
              pk_id: NOT_YET_PUSHED_ID,
              labelId: triggerIndex,
              base_ann_id: this.$store.getters["explanation/getAnnotationInfo"].extended_annotation.annotation_id,
            });
          }
        }

        console.log("triggers 11111", this.triggers)

      }
    },
    // send explanations to the backend. will first delete the removed explanation, then add the new un-submitted explanation
    submitExplanations() {
      console.log("submit", this.triggers)
      const promisesToDelete = [];
      const promisesToAdd = [];

      const projectId = this.$store.getters.getProjectInfo.id;
      const documentId = this.$store.getters["document/getCurDoc"].id;
      const annotationId = this.$store.getters["explanation/getAnnotationInfo"].id;

      this.reasonsToDelete.forEach((reasonID) => {
        promisesToDelete.push(
            this.$http.delete(`/projects/${projectId}/docs/${documentId}/annotations/${annotationId}/trigger/${reasonID}`)
                .catch(err => console.log("failed to delete", err))
        )
      })
      Promise.all(promisesToDelete).then(() => {
        this.triggers
            .forEach((triggers, triggerIndex) => {
              console.log("trigger", triggers, triggerIndex);
              triggers.filter(reason => reason.pk_id === NOT_YET_PUSHED_ID).filter(reason => reason.start_offset != -1 && reason.end_offset != -1).forEach(reason => {
                console.log("reason", reason);
                let addPromise = this.$http.post(`/projects/${projectId}/docs/${documentId}/annotations/${annotationId}/trigger/`,
                    {start_offset: reason.start_offset, end_offset: reason.end_offset, trigger_id: triggerIndex})
                promisesToAdd.push(addPromise);
              })
            })
        return Promise.all(promisesToAdd);
      }).then(() => {
        this.$store.dispatch("document/fetchDocuments");
        this.$store.dispatch('explanation/hideExplanationPopup');
        this.$notify({type: "success", message: "Explanations updated", title: "Success"})
      }).catch(err => {
        console.error("failed to add explanations", err)
      });


    },
    // delete trigger group
    removeTrigger(index) {
      console.log("remove trigger", this.triggers, this.triggers[index])
      this.triggers[index].filter(reason => reason.pk_id > 0).forEach(reason => this.reasonsToDelete.push(reason.pk_id))
      this.triggers.splice(index, 1);
      if (this.triggers.length === 0) {
        this.addAnotherExplanation();
      }
    },
    // delete reason
    removeReason(triggerIndex, chunkIndex, chunk) {
      const reasonIndex = this.triggers[triggerIndex].findIndex(reason => reason.pk_id === chunk.pk_id);
      this.triggers[triggerIndex].splice(reasonIndex, 1).filter(reason => reason.pk_id > 0).forEach(reason => this.reasonsToDelete.push(reason.pk_id));
    },
    // get the css style for the chunk
    getChunkStyle(chunk, labelId) {
      // Check if the chunk matches the current label ID. If so, we want to
      // highlight the text span.
      if (chunk.labelId === labelId) {
        return {
          color: this.id2label[labelId].text_color,
          backgroundColor: this.id2label[labelId].background_color,
          textDecoration: this.id2label[labelId].text_decoration,
          padding: '10px 10px 10px 10px', fontSize: "18px",
          borderRadius: "8px"
        };
      }
      // If the chunk is a label but doesn't match our current label ID, we want
      // to mark it as grey to discourage users from selecting again.
      if (chunk.labelId !== UNLABELED_LABEL_ID) {
        return {
          color: "#bfbfbf",
        };
      }
      // Return empty style for a chunk that isn't a label.
      return {};
    },
    // get the style for trigger
    generateTriggerLabels() {
      const triggerBackgroundColors = ["#95de64", "#5cdbd3", "#85a5ff", "#b37feb"];
      triggerBackgroundColors.forEach((color, index) => {
        this.triggerLabels.push({
          id: index,
          text: index,
          text_color: "white",
          background_color: color,
          text_decoration: "",
          shortcut: index,
        })
      })

    }
  },
  computed: {
    dialogVisible: {
      get() {
        return this.$store.getters["explanation/getExplanationPopupInfo"].dialogVisible;
      },
      set() {
        this.$store.dispatch("explanation/hideExplanationPopup")
      }
    },
    // generate chunks to assemble the full text
    chunks() {
      const res = [];
      let left = 0;
      if (!this.$store.getters["explanation/getAnnotationInfo"]) {
        return res;
      }
      // const triggers = this.$store.getters["explanation/getAnnotationInfo"].formattedTriggerExplanation;
      console.log("triggers", this.triggers);

      const allTriggers = this.triggers.flat();
      // console.log("all triggers", allTriggers);

      allTriggers.sort((a, b) => a.start_offset - b.start_offset);

      for (const l of allTriggers) {
        if (l.start_offset === -1 || l.end_offset === -1){
          continue;
        }
        res.push({
          pk_id: UNLABELED_LABEL_ID,
          start_offset: left,
          end_offset: l.start_offset,
          labelId: UNLABELED_LABEL_ID,
        });

        res.push({
          pk_id: l.pk_id,
          start_offset: l.start_offset,
          end_offset: l.end_offset,
          labelId: l.labelId,
        });

        left = l.end_offset;
      }

      res.push({
        pk_id: UNLABELED_LABEL_ID,
        start_offset: left,
        end_offset: this.$store.getters["document/getCurDoc"].text.length,
        labelId: UNLABELED_LABEL_ID,
      });
      console.log("chunks", res);
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

      for (let i = 0; i < this.triggerLabels.length; i++) {
        const label = this.triggerLabels[i];
        label.text_decoration = "";
        id2label[label.id] = label;
      }
      return id2label;
    },
    canAddExplanation() {
      const lastTrigger = this.triggers[this.triggers.length - 1];
      if (!lastTrigger) {
        return false;
      }
      const lastReason = lastTrigger[lastTrigger.length - 1];
      console.log("last reason", lastReason)
      if (!lastReason || lastReason.start_offset === -1 || lastReason.end_offset === -1) {
        console.log("can not add")
        return false;
      } else {
        return true;
      }
    }
  },

}
</script>

<style scoped>

</style>
