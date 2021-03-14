<template>
  <el-card style="text-align: left; margin-top: 20px">
    <el-row>
      <el-tag>Text</el-tag>
    </el-row>
    <el-col style="line-height: 4; margin-top: 10px;" v-if="$store.getters['document/getCurDoc']">
      {{ this.$store.getters["document/getCurDoc"].text }}
    </el-col>

    <SelectedLabelListRow/>
  </el-card>

</template>

<script>
import SelectedLabelListRow from "@/components/project/annotation/sa/SelectedLabelListRow";
import {ACTION_TYPE, DIALOG_TYPE} from "@/utilities/constant";

export default {
  name: "SentimentAnalysisAnnotation",
  components: {SelectedLabelListRow},
  data() {
    return {}
  },
  created() {
    if (this.$store.getters.getActionType === ACTION_TYPE.CREATE) {
      this.$store.commit("showAnnotationGuidePopup", DIALOG_TYPE.Annotation.SA);
    }
  }
}
</script>

<style scoped>

</style>
