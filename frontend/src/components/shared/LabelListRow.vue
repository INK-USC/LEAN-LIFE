<template>
  <el-col :span="16" style="display: flex; flex-wrap: wrap">
    <Label v-for="label of this.$store.getters['label/getLabels']" :key="label.id" :labelInfo="label"
           style="margin-right: 10px"/>
  </el-col>
</template>

<script>
import Label from "@/components/shared/Label";

// put label in a row
export default {
  name: "LabelListRow",
  components: {Label},
  created() {
    this.$store.dispatch('label/fetchLabels', null, {root: true})
  }

}
</script>

<style scoped>

</style>
