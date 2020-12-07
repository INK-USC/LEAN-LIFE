<template>
  <el-row>
    <h1>Annotate page</h1>

    <el-progress :percentage="50" :format="getProgressBarLabel" type="line" :stroke-width="30" text-inside/>
    <el-row style="margin-top: 20px">
      <LabelListRow/>
    </el-row>

    <SentimentAnalysisAnnotation/>

    <el-row style="margin-top: 20px">
      <el-col style="display: flex; justify-content: space-between">
        <el-button icon="el-icon-arrow-left" type="primary">Prev</el-button>
        <el-button type="primary">Next <i class="el-icon-arrow-right"/></el-button>
      </el-col>
    </el-row>
    <el-row style="margin-top: 10px">
      <el-col style="display: flex; justify-content: flex-end">
        <el-button type="primary">Skip (Nothing to Mark Up)<i class="el-icon-arrow-right"/></el-button>
      </el-col>
    </el-row>
  </el-row>
</template>

<script>
import LabelListRow from "@/components/shared/LabelListRow";
import SentimentAnalysisAnnotation from "@/components/project/annotation/sa/SentimentAnalysisAnnotation";

export default {
  name: "Annotate",
  components: {SentimentAnalysisAnnotation, LabelListRow},
  data() {
    return {}
  },
  methods: {
    getProgressBarLabel() {
      return "0/10 to annotate"
    }
  }
}
</script>

<style scoped>

</style>
