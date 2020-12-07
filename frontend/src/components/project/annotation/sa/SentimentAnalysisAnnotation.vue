<template>
  <el-card style="text-align: left; margin-top: 20px">
    <el-row>
      <el-tag>Text</el-tag>
    </el-row>
    <el-row style="line-height: 4; margin-top: 10px;" v-if="$store.getters['document/getDocuments'].documents">
      {{
        this.$store.getters["document/getDocuments"]
            .documents[this.$store.getters["document/getDocuments"].curDocIndex]['text']
      }}
    </el-row>
    <el-divider/>
    <el-row>
      <el-tag>Selected Labels</el-tag>
    </el-row>
  </el-card>

</template>

<script>
export default {
  name: "SentimentAnalysisAnnotation",
  data() {
    return {
      document: {}
    }
  }
}
</script>

<style scoped>

</style>
