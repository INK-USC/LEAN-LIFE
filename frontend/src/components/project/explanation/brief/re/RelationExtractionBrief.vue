<template>
  <div>
    {{ this.$store.getters["document/getCurDoc"].text }}
    <span v-if="this.currentRelation">
      <RelationDisplay :relation="this.currentRelation" :hideDelete="true"/>
    </span>

  </div>
</template>

<script>
import RelationDisplay from "@/components/project/annotation/re/RelationDisplay";
// show a simplified version of RE document.

export default {
  name: "RelationExtractionBrief",
  components: {RelationDisplay},
  computed: {
    currentRelation() {
      if (!this.$store.getters['document/getCurDoc']){
        return;
      }
      const selection = this.$store.getters['annotation/getRESelection'];
      const relation = this.$store.getters['document/getCurDoc'].formattedAnnotations.find(
        annotation => (
          annotation.obj_end_offset === selection.obj_end_offset &&
          annotation.obj_start_offset === selection.obj_start_offset &&
          annotation.sbj_end_offset === selection.sbj_end_offset &&
          annotation.sbj_start_offset === selection.sbj_start_offset
        )
      )
      return relation;
    }
  }
}
</script>

<style scoped>

</style>
