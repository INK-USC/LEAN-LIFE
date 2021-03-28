<template>
  <div>
    <!--    <el-divider/>-->
    <el-row style="margin-top: 20px">
      <el-tag>Selected Labels</el-tag>
    </el-row>
    <el-row style="margin-top: 10px">
      <div v-if="!this.$store.getters['document/getCurDoc'] || !this.$store.getters['label/getLabels']"/>
      <SelectedLabel
          v-else
          v-for="annotationInfo of this.$store.getters['document/getCurDoc'].annotations"
          :key="annotationInfo.id" :annotationInfo="annotationInfo" style="margin-right: 10px"
      />
    </el-row>
  </div>


</template>

<script>

import SelectedLabel from "@/components/project/annotation/sa/SelectedLabel";

export default {
  name: "SelectedLabelListRow",
  components: {SelectedLabel},
  created() {
  }

}
</script>

<style scoped>

</style>
