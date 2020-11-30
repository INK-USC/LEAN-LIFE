<template>
  <el-card>
    <div slot="header"><h3>Annotation Settings</h3></div>
    <div style="text-align: left">
      <el-row v-if="this.$store.getters.getProjectInfo.task!==1">
        <el-col :span="8">
          <el-tooltip
              content="Recommendation options to help in the annotation process, multiple options can be selected at once.">
            <b>Recommendations</b>
          </el-tooltip>
        </el-col>
        <el-col :span="16">
          <el-tooltip content="Highlight Noun Phrases as possible Named Entity Spans"
                      v-if="this.$store.getters.getProjectInfo.task==2">
            <el-checkbox v-model="annotationSettings.noun_chunk" label="Noun Chunk" border/>
          </el-tooltip>
          <el-tooltip>
            <span slot="content">
              Previous annotations will be cached and applied when appropriate as recommendations.
              <br/>
              You can upload your own list of annotations as an initial seed on the Historical page
              <br/>
              (accessible by checking this box and clicking the button below)
            </span>
            <el-checkbox v-model="annotationSettings.history" label="Historical" border/>
          </el-tooltip>
        </el-col>
      </el-row>
      <el-row style="margin-top: 20px;">
        <el-col :span="8">
          <el-tooltip
              content="Number of documents that should be retrieved for annotations when the annotator has completed current batch.">
            <b>Acquire size</b>
          </el-tooltip>
        </el-col>
        <el-col :span="16">
          <el-input-number v-model="annotationSettings.acquire"/>
        </el-col>
      </el-row>
      <el-divider/>

      <el-row>
        <el-button type="primary" @click="saveSettings">SAVE</el-button>
        <el-button type="danger" @click="resetSettings">REST</el-button>
        <el-button type="warning" @click="goToHistoricalAnnotations" v-if="annotationSettings.history">Upload Historical
          Annotations
        </el-button>
      </el-row>
    </div>
  </el-card>
</template>

<script>
export default {
  name: "AnnotationSettings",
  data() {
    return {
      annotationSettings: {}
    }
  },
  methods: {
    fetchAnnotationSettings() {
      return this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/settings/`)
          .then(res => {
            console.log("setting ", res)
            this.annotationSettings = res;
          })
    },
    saveSettings() {
      return this.$http
          .put(`/projects/${this.$store.getters.getProjectInfo.id}/settings/`, this.annotationSettings)
          .then(res => {
            console.log("setting saved", res)
            this.annotationSettings = res;
          })
    },
    resetSettings() {
      this.fetchAnnotationSettings()
    },
    goToHistoricalAnnotations() {
      this.$router.push({name: "HistoricalAnnotations"})
    }
  },
  created() {
    this.fetchAnnotationSettings()
        .catch(err => {
          return this.saveSettings()
        })
        .finally(() => {
          this.fetchAnnotationSettings()
        })
  }
}
</script>

<style scoped>

</style>
