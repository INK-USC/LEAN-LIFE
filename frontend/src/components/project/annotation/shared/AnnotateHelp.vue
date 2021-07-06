<template>
  <el-popover trigger="hover" content="Click to show how to annotate document">
    <span slot="reference" style="margin-right: 10px">
        <el-button icon="el-icon-question" @click="showAnnotationGuidePopup">
          Help
        </el-button>
    </span>
  </el-popover>
</template>

<script>
import {DIALOG_TYPE} from "@/utilities/constant";

export default {
  name: "AnnotateHelp",
  methods: {
    showAnnotationGuidePopup() {
      this.$store.commit("showAnnotationGuidePopup", this.projectType);
    }
  },
  computed: {
    projectType() {
      const task = this.$store.getters.getProjectInfo.task;
      if (task === 1) {
        return DIALOG_TYPE.Annotation.SA;
      } else if (task === 2) {
        return DIALOG_TYPE.Annotation.NER;
      } else if (task === 3) {
        return DIALOG_TYPE.Annotation.RE;
      } else {
        return Error;
      }
    }
  }
}
</script>

<style scoped>

</style>
