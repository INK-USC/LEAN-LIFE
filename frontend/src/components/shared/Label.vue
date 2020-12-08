<template>
  <div style="display: flex; align-items: center">
    <el-tag class="label-tag" :style="{backgroundColor: labelInfo.background_color, color: labelInfo.text_color}"
            style="display: flex; align-items: baseline">
      <b style="font-size: medium" @click="addLabel"
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
    addLabel() {
      if (!this.$route.path.endsWith('annotate')) {
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
            return this.$http
                .post(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${lastAnnotationId}/sa/`, {})
          })
          .then(() => {
            console.log("sa post completed")
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
    }
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
