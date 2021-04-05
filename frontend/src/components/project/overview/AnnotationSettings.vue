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
        <el-button type="primary" @click="saveSettings(true)">SAVE</el-button>
        <el-button type="danger" @click="resetSettings">REST</el-button>
        <el-button type="warning" @click="goToHistoricalAnnotations" v-if="annotationSettings.history">Upload Historical
          Annotations
        </el-button>
      </el-row>
    </div>
  </el-card>
</template>

<script>
import {ACTION_TYPE, DIALOG_TYPE} from "@/utilities/constant";

// settings for annotations.
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
    saveSettings(goToNextStep) {
      return this.$http
          .put(`/projects/${this.$store.getters.getProjectInfo.id}/settings/`, this.annotationSettings)
          .then(res => {
            console.log("setting saved", res)
            this.annotationSettings = res;
            if (goToNextStep) {
              this.goNextStep();
            }
          })
    },
    resetSettings() {
      this.fetchAnnotationSettings()
    },
    goToHistoricalAnnotations() {
      this.$router.push({name: "HistoricalAnnotations"})
    },
    goNextStep() {
      this.$store.commit('updateActionRelatedInfo', {step: 4});
      this.$router.push({name: "Annotate"})
    }
  },
  // on creation, first fetch settings, will get error, then manually save default setting, then fetch again to avoid blank default setting on page
  created() {
    if (this.$store.getters.getActionType === ACTION_TYPE.CREATE) {
      this.$store.commit("showSimplePopup", DIALOG_TYPE.ConfiguringOptionalAnnotationSettings);
    }
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
