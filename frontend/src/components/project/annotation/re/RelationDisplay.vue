<template>
  <div style="line-height: 4; display: flex; justify-content: space-between; align-items: center"
       @click="showExplanationPopup">
    <div>
      <span style="margin-right: 10px" class="word">{{ sbjText }}</span>

      <div class="relation-wrapper">
        <div class="relation-text-wrapper">
          <span :style="{display: 'inline-block', color: labelInfo.background_color}"> {{ labelInfo.text }}</span>
        </div>
        <div class="arrow-wrapper">
          <div class="arrow-line" :style="{ background: labelInfo.background_color }"></div>
          <div class="arrow-head" :style="{ borderLeftColor: labelInfo.background_color }"></div>
        </div>
      </div>

      <span class="word">{{ objText }}</span>
    </div>

    <i class="el-icon-delete" style="cursor: pointer" @click="removeSelectedRelation"/>
  </div>
</template>

<script>
// display relation for subject and object.
export default {
  name: "RelationDisplay",
  props: {
    relation: Object
  },
  methods: {
    // delete selected relation when user clicked
    removeSelectedRelation() {
      const docId = this.$store.getters["document/getCurDoc"].id
      console.log("id", docId)
      this.$http.delete(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${docId}/annotations/${this.relation.base_ann_id}`)
          .then(() => {
            if (this.$store.getters["document/getCurDoc"].formattedAnnotations.filter(annotation => annotation.type === "re").length == 1) {
              return this.$http.patch(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${docId}`, {annotated: false})
                  .then(() => {
                  })
            }
          }).then(() => {
        this.$store.dispatch("document/fetchDocuments", {})
      })
    },
    // user can add explanation
    showExplanationPopup() {
      console.log("relation", this.relation)
      this.$store.dispatch("annotation/setRESelection", {
        objStart: this.relation.obj_start_offset,
        objEnd: this.relation.obj_end_offset,
        sbjStart: this.relation.sbj_start_offset,
        sbjEnd: this.relation.sbj_end_offset,
        objText: this.objText,
        sbjText: this.sbjText,
      }).then(() => {
        this.$store.dispatch("explanation/showExplanationPopup", {annotationId: this.relation.base_ann_id})
      })
    }
  },
  computed: {
    // get subject text
    sbjText() {
      return this.$store.getters["document/getCurDoc"].text.slice(this.relation.sbj_start_offset, this.relation.sbj_end_offset)
    },
    // get object text
    objText() {
      return this.$store.getters["document/getCurDoc"].text.slice(this.relation.obj_start_offset, this.relation.obj_end_offset)
    },
    // get all the label info
    labelInfo() {
      let label = this.$store.getters["label/getLabels"].find(label => label.id === this.relation.label)
      return label;
    }
  }
}
</script>

<style scoped>
.relation-wrapper {
  display: inline-block;
  padding-right: 10px;
}

.relation-text-wrapper {
  position: absolute;
  width: 93px;
  text-align: center;
}

.relation-text-wrapper > span {
  position: relative;
  top: -15px;
  font-size: 14px;
}

.arrow-wrapper {
  width: 93px;
  display: inline-block;
}

.arrow-line {
  margin-top: 7px;
  width: 80px;
  height: 3px;
  float: left;
}

.arrow-head {
  width: 0;
  height: 0;
  border-top: 8px solid transparent;
  border-bottom: 8px solid transparent;
  border-left-width: 13px;
  border-left-style: solid;
  float: right;
}

.word {
  padding: 0.4em;
  background-color: #f5f5f5;
  border-radius: 2px;
}
</style>
