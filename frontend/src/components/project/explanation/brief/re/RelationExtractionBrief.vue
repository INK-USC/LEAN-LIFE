<template>
  <div>
    {{ this.$store.getters["document/getCurDoc"].text }}
    <span v-if="this.$store.getters['document/getCurDoc']">
      <RelationDisplay
          v-for="(relation, index) in this.$store.getters['document/getCurDoc'].formattedAnnotations.filter(annotation=>annotation.type==='re')"
          :key="index" :relation="relation"/>
    </span>

  </div>
</template>

<script>
import RelationDisplay from "@/components/project/annotation/re/RelationDisplay";
// show a simplified version of RE document.

export default {
  name: "RelationExtractionBrief",
  components: {RelationDisplay}
}
</script>

<style scoped>

</style>
