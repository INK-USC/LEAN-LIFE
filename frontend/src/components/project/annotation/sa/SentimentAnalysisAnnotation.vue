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

export default {
  name: "SentimentAnalysisAnnotation",
  components: {SelectedLabelListRow},
  data() {
    return {}
  }
}
</script>

<style scoped>

</style>
