<template>
  <el-row>
    <h1>
      Annotate page
      <span v-if="this.$store.getters.getProjectInfo.task==1">(Sentiment Analysis)</span>
      <span v-if="this.$store.getters.getProjectInfo.task==2">(Named Entity Recognition)</span>
      <span v-if="this.$store.getters.getProjectInfo.task==3">(Relation Extraction)</span>
    </h1>

    <el-progress :percentage="annotationCompletionPercentage"
                 :format="getProgressBarLabel" type="line" :stroke-width="30" text-inside/>
    <el-row style="margin-top: 20px">
      <LabelListRow/>
    </el-row>

    <SentimentAnalysisAnnotation v-if="this.$store.getters.getProjectInfo.task==1"/>
    <NamedEntityRecognitionAnnotation v-if="this.$store.getters.getProjectInfo.task==2"/>
    <RelationExtractionAnnotation v-if="this.$store.getters.getProjectInfo.task==3"/>

    <el-row style="margin-top: 20px">
      <el-col style="display: flex; justify-content: space-between">
        <el-button icon="el-icon-arrow-left" type="primary" @click="goToNextDoc(false)"
                   :disabled="this.$store.getters['document/getDocuments'].curDocIndex===0
                   && this.$store.getters['document/getDocuments'].curPage===1">
          Prev
        </el-button>
        <el-button type="primary" @click="goToNextDoc(true)"
                   :disabled="this.$store.getters['document/getDocuments'].curDocIndex===this.$store.getters['document/getDocuments'].documents.length-1
                   && this.$store.getters['document/getDocuments'].curPage === this.$store.getters['document/getDocuments'].maxPage">
          Next <i class="el-icon-arrow-right"/>
        </el-button>
      </el-col>
    </el-row>
    <el-row style="margin-top: 10px">
      <el-col style="display: flex; justify-content: flex-end">
        <el-button type="primary"
                   @click="noAnnotation"
                   :disabled="this.$store.getters['document/getDocuments'].curDocIndex===this.$store.getters['document/getDocuments'].documents.length-1
                   && this.$store.getters['document/getDocuments'].curPage === this.$store.getters['document/getDocuments'].maxPage">
          Skip (Nothing to Mark Up)<i class="el-icon-arrow-right"/>
        </el-button>
      </el-col>
    </el-row>
  </el-row>
</template>

<script>
import LabelListRow from "@/components/shared/LabelListRow";
import SentimentAnalysisAnnotation from "@/components/project/annotation/sa/SentimentAnalysisAnnotation";
import NamedEntityRecognitionAnnotation from "@/components/project/annotation/ner/NamedEntityRecognitionAnnotation";
import RelationExtractionAnnotation from "@/components/project/annotation/re/RelationExtractionAnnotation";

export default {
  name: "Annotate",
  components: {
    RelationExtractionAnnotation,
    NamedEntityRecognitionAnnotation,
    SentimentAnalysisAnnotation,
    LabelListRow
  },
  data() {
    return {}
  },
  methods: {
    getProgressBarLabel() {
      let documentInfo = this.$store.getters["document/getDocuments"];
      return `${documentInfo.annotatedDocCount} / ${documentInfo.totalDocCount} documents annotated`
    },
    goToNextDoc(isNext) {
      let curDocIndex = this.$store.getters['document/getDocuments'].curDocIndex;
      this.$store.dispatch('document/updateCurDocIndex',
          {curDocIndex: isNext ? curDocIndex + 1 : curDocIndex - 1},
          {root: true})
    },
    noAnnotation() {
      this.$http
          .patch(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}`,
              {annotated: true})
          .then(() => {
            this.$store.dispatch('document/fetchDocuments')
          })
      this.goToNextDoc(true)
    }
  },
  computed: {
    annotationCompletionPercentage: function () {
      let documentInfo = this.$store.getters["document/getDocuments"];
      if (documentInfo.totalDocCount == 0) {
        return 0;
      } else {
        let percentage = 100 * documentInfo.annotatedDocCount / documentInfo.totalDocCount;
        return percentage
      }
    },
  }
}
</script>

<style scoped>

</style>
