<template>
  <div style="display: flex; align-items: center">
    <el-tag class="label-tag" :style="getColorStyle" style="display: flex; align-items: baseline">
      <b style="font-size: medium" @click="addAnnotation"
         :style="getCursorStyle">{{ labelInfo.text }}</b>
      <i class="el-icon-close el-icon--right" @click="removeLabel"
         v-if="!this.$route.path.endsWith('annotate') && this.labelInfo.text"/>
    </el-tag>
    <el-tag class="label-keyboard-shortcut" style="background-color: white">
      <b>{{ labelInfo.shortcut | displayShortcut }}</b>
    </el-tag>
  </div>
</template>

<script>
export default {
  name: "Label",
  props: {labelInfo: Object},
  methods: {
    removeLabel() {
      this.$http.delete(`/projects/${this.$store.getters.getProjectInfo.id}/labels/${this.labelInfo.id}`).then(res => {
        this.$store.dispatch('label/fetchLabels', null, {root: true})
      })
    },
    addAnnotation() {
      if (!this.$route.path.endsWith('annotate')) {
        return;
      }
      if (!this.canClick) {
        if (this.$store.getters.getProjectInfo.task === 1) {
          this.$notify({
            type: "warning", title: "Warning",
            message: "This document has already been annotated with this label!"
          })
        }
        if (this.$store.getters.getProjectInfo.task === 2) {
          this.$notify({type: "warning", title: "Warning", message: "Please select words first!"})
        }
        if (this.$store.getters.getProjectInfo.task === 3) {
          this.$notify({type: "warning", title: "Warning", message: "Please select subject and object first!"})
        }
        return;
      }
      //NER
      if (this.$store.getters.getProjectInfo.task == 2 && (this.$store.getters["annotation/getNERSelection"].selectionStart == -1 || this.$store.getters["annotation/getNERSelection"].selectionEnd == -1)) {
        this.$notify({
          type: "error",
          message: "invalid selection" + this.$store.getters["annotation/getNERSelection"],
          title: "annotate failed"
        })
        return;
      }
      let annotationId = -1;
      this.$http
          .post(
              `/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/`,
              {
                label: this.labelInfo.id,
                via_recommendation: false
              }
          )
          .then(res => {
            console.log("label added", res)

            let lastAnnotationId = res.id;
            annotationId = res.id

            if (this.$store.getters.getProjectInfo.task == 1) {
              return this.$http
                  .post(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${lastAnnotationId}/sa/`, {})
            } else if (this.$store.getters.getProjectInfo.task == 2) {
              let nerData = {
                "start_offset": this.$store.getters["annotation/getNERSelection"].selectionStart,
                "end_offset": this.$store.getters["annotation/getNERSelection"].selectionEnd,
                "user_provided": false,
              }
              return this.$http
                  .post(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${lastAnnotationId}/ner/`, nerData)
            } else if (this.$store.getters.getProjectInfo.task == 3) {
              //TODO
              let reData = this.$store.getters["annotation/getRESelection"];
              return this.$http
                  .post(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${lastAnnotationId}/re/`, reData)
            }

          })
          .then(() => {
            console.log("post completed")
            return this.$http
                .patch(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}`,
                    {annotated: true})
          })
          .then(() => {
            console.log("patch completed")
            if (this.$store.getters.getProjectInfo.task === 3) {
              return this.$http
                  .post(`/projects/${this.$store.getters.getProjectInfo.id}/history/re/`, {
                    annotation: annotationId,
                    label: this.labelInfo.id,
                    user_provided: false,
                    word_1: this.$store.getters["annotation/getRESelection"].sbj_text,
                    word_2: this.$store.getters['annotation/getRESelection'].obj_text,
                  })
            }
          })
          .then(() => {
            this.$store.dispatch('document/fetchDocuments').then(() => {
              this.$store.dispatch("explanation/showExplanationPopup", {annotationId: annotationId})
            })
          })
          .catch(err => {
            console.log("err", err)
          })
    },

  },
  created() {
  },
  computed: {
    canClick() {
      console.log("curdoc", this.$store.getters["document/getCurDoc"])
      if (!this.$store.getters["document/getCurDoc"]) {
        return false;
      }

      if (this.$store.getters.getProjectInfo.task === 1) {
        //SA
        const annotations = this.$store.getters["document/getCurDoc"].annotations;
        const canClick = !annotations.some(ann => ann.label === this.labelInfo.id)
        console.log("can click", canClick)
        return canClick;
      } else if (this.$store.getters.getProjectInfo.task === 2) {
        //NER
        const nerSelection = this.$store.getters["annotation/getNERSelection"]
        console.log("ner selection ", nerSelection)
        if (nerSelection.selectionStart === -1 || nerSelection.selectionEnd === -1) {
          console.log("can not click")
          return false;
        } else {
          console.log("can click")
          return true;
        }
      } else if (this.$store.getters.getProjectInfo.task === 3) {
        //RE
        const reSelection = this.$store.getters["annotation/getRESelection"]
        console.log("re selection ", reSelection)
        for (let key in reSelection) {
          if (reSelection[key] === -1) {
            return false;
          }
        }
        return true;
      }
      return 1111;
    },
    getCursorStyle() {
      if (this.$route.path.endsWith("annotate")) {
        if (this.canClick) {
          return {cursor: "pointer"};
        } else {
          return {cursor: "not-allowed"};
        }
      } else {
        return {cursor: "default"};
      }
    },
    getColorStyle() {
      if (this.$route.path.endsWith("annotate") && !this.canClick) {
        return {backgroundColor: 'grey', color: 'white'};
      } else {
        return {backgroundColor: this.labelInfo.background_color, color: this.labelInfo.text_color};
      }
    }
  }

}
</script>

<style scoped>
.label-tag {
  border-bottom-right-radius: 0px;
  border-top-right-radius: 0px;
}

.label-keyboard-shortcut {
  background: #f5f5f5;
  border-bottom-left-radius: 0px;
  border-top-left-radius: 0px
}

</style>
