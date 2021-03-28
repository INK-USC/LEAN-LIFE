<template>
  <div>
    <el-tag :style="{backgroundColor: this.$store.getters['explanation/getSelectedLabelInfo'].background_color,
          color: this.$store.getters['explanation/getSelectedLabelInfo'].text_color, fontSize: 'medium'
        }">
      <b>{{ this.$store.getters["explanation/getSelectedLabelInfo"].text }}</b>
    </el-tag>
    <el-row style="line-height: 2; font-size: medium; margin-top: 15px">
      {{ this.$store.getters["document/getCurDoc"].text }}
    </el-row>
  </div>
</template>

<script>
export default {
  name: "SentimentAnalysisBrief"
}
</script>

<style scoped>

</style>
