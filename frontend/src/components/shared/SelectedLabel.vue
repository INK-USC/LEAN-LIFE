<template>
  <el-tag v-if="!!labelInfo"
          :style="{backgroundColor: fullInfo.labelInfo.background_color, color: fullInfo.labelInfo.text_color}">

    <b style="font-size: medium">{{ this.labelText }}</b>
    <i class="el-icon-close el-icon-right" @click="detachLabelFromDocs"/>
  </el-tag>
</template>

<script>
export default {
  name: "SelectedLabel",
  props: {labelInfo: Object},
  data() {
    return {
      fullInfo: {}
    }
  },
  methods: {
    detachLabelFromDocs() {
      console.log("need to detach from docs")
      this.$http
          .delete(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/annotations/${this.fullInfo.annotationInfo.id}`)
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
    }
  },
  created() {
    let res = this.$store.getters["label/getLabels"].find(label => {
      return label.id === this.labelInfo.label;
    })
    this.fullInfo = {annotationInfo: this.labelInfo, labelInfo: res}
    // console.log("combined ", this.fullInfo)
  },
  computed: {
    labelText: function () {
      let res = this.$store.getters["label/getLabels"].find(label => {
        return label.id === this.labelInfo.label;
      })
      return res.text
    }
  }
}
</script>

<style scoped>

</style>
