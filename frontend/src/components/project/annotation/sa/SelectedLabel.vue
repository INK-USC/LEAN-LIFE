<template>
  <el-tag v-if="!!label"
          :style="{backgroundColor: label.background_color, color: label.text_color}"
  >
    <b style="font-size: medium; cursor: pointer" @click="showExplanationPopup">{{ label.text }}</b>
    <i class="el-icon-close el-icon-right" @click="removeAnnotation"/>
  </el-tag>

</template>

<script>
// show all the selected label for current doc
export default {
  name: "SelectedLabel",
  props: {annotationInfo: Object},
  methods: {
    // remove annotation for current doc
    removeAnnotation() {
      this.$http
          .delete(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${this.annotationInfo.id}`)
          .then(res => {
            console.log("response for delete", res);
            if (this.$store.getters["document/getCurDoc"].annotations.length == 1) {
              return this.$http
                  .patch(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}`,
                      {annotated: false}
                  )
            }
          })
          .then(() => {
            console.log("patched")
            this.$store.dispatch('document/fetchDocuments')
          })
    },
    // show the explanation popup
    showExplanationPopup() {
      this.$store.dispatch("explanation/showExplanationPopup", {annotationId: this.annotationInfo.id})
    }
  },
  computed: {
    label() {
      return this.$store.getters["label/getLabels"].find(label => label.id === this.annotationInfo.label)
    }
  }
}
</script>

<style scoped>

</style>
