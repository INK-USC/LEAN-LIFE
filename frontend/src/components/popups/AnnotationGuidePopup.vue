<template>
  <el-dialog :visible.sync="dialogVisible">
    <h1 slot="title" style="text-align: center">
      <i class="el-icon-info"/>
      {{ this.$store.getters.getAnnotationGuidePopupInfo.targetDialogType }} Annotation Guideline
    </h1>
    <div style="text-align: center">
      <div style="font-size: 20px">Here's a quick animation on the different ways you can annotate a document</div>
      <img style="margin-top: 20px" :src="getImg()"/>
    </div>
    <span slot="footer">
      <el-button type="primary" @click="()=>this.$store.commit('hideAnnotationGuidePopup')">OK</el-button>
    </span>
  </el-dialog>
</template>

<script>
// guide on how to annotate. will popup when user first go to annotation page. it will be different for SA, NER, RE. only show when they first create project
export default {
  name: "AnnotationGuidePopup",
  data() {
    return {}
  },
  methods: {
    getImg() {
      if (this.$store.getters.getAnnotationGuidePopupInfo.targetDialogType !== ""){
        return require(`@/assets/annotation_gif/${this.$store.getters.getAnnotationGuidePopupInfo.targetDialogType}_Annotation.gif`);
      }
    }
  },
  computed: {
    dialogVisible: {
      get() {
        return this.$store.getters.getAnnotationGuidePopupInfo.dialogVisible
      },
      set() {
        this.$store.commit("hideAnnotationGuidePopup")
      }
    },
  }
}
</script>

<style scoped>

</style>
