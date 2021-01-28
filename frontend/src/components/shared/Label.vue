<template>
  <div style="display: flex; align-items: center">
    <el-tag class="label-tag" :style="{backgroundColor: labelInfo.background_color, color: labelInfo.text_color}"
            style="display: flex; align-items: baseline">
      <b style="font-size: medium" @click="addAnnotation"
         :style="{cursor: this.$route.path.endsWith('annotate')? 'pointer': 'default'}">{{ labelInfo.text }}</b>
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
      //NER
      if (this.$store.getters.getProjectInfo.task == 2 && this.$store.getters["annotation/getNERSelection"].start_offset == -1 || this.$store.getters["annotation/getNERSelection"].end_offset == -1) {
        return;
      }
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
              // return this.$http
              //     .post(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${lastAnnotationId}/re/`, {})
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
            this.$store.dispatch('document/fetchDocuments')
          })
          .catch(err => {
            console.log("err", err)
          })
    },

  },
  created() {
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
