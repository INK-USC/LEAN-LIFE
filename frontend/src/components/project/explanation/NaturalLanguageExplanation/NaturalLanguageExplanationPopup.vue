<template>
  <el-dialog :visible.sync="dialogVisible" v-if="!!this.$store.getters['document/getCurDoc']"
             class="explanation-dialog" @open="dialogOpen">
    <h1 slot="title">
      Natural Language Explanation
    </h1>
    <div style="text-align:initial">
      <div>
        Please select a template below and:
      </div>
      <ol>
        <li>fill in the blanks</li>
        <li>decide between the two options in the '[]'</li>
      </ol>
      <div>
        You may also type out your own explanations, but please do so in a way similar to the provided templates.
      </div>

      <el-divider/>

      <SentimentAnalysisBrief v-if="this.$store.getters.getProjectInfo.task===1"/>
      <NamedEntityRecognitionBrief v-if="this.$store.getters.getProjectInfo.task===2"/>
      <RelationExtractionBrief v-if="this.$store.getters.getProjectInfo.task===3"/>

      <el-form style="margin-top: 20px">
        <el-form-item v-for="(reason, index) in this.reasons" :key=index>
          <div style="display: flex; flex-direction: row; align-items: center; justify-content: space-between">
            <el-autocomplete v-model="reasons[index].text" placeholder="Please explain why"
                             :fetch-suggestions="searchExplanationTemplate" style="width: 98%" clearable/>
            <el-button icon="el-icon-delete" @click="deleteExplanation(index)" :disabled="reasons.length==1"/>
          </div>
        </el-form-item>
        <el-form-item>
          <el-button @click="addAnotherExplanation"
                     style="width: 100%; border: 1px dotted" type="primary"
                     :disabled="this.reasons[this.reasons.length-1].text.trim()===''">
            <i class="el-icon-plus"/>
            Add additional explanation
          </el-button>
        </el-form-item>
      </el-form>
    </div>
    <span slot="footer">
      <el-button @click="$store.dispatch('explanation/hideExplanationPopup')">Cancel</el-button>
      <el-button type="primary" @click="submitExplanations" :loading="buttonIsLoading">OK</el-button>
    </span>
  </el-dialog>
</template>

<script>
import SentimentAnalysisBrief from "@/components/project/explanation/brief/sa/SentimentAnalysisBrief";
import NamedEntityRecognitionBrief
  from "@/components/project/explanation/brief/ner/NamedEntityRecognitionBrief";
import RelationExtractionBrief from "@/components/project/explanation/brief/re/RelationExtractionBrief";


export default {
  name: "NaturalLanguageExplanationPopup",
  components: {RelationExtractionBrief, NamedEntityRecognitionBrief, SentimentAnalysisBrief},

  data() {
    return {
      reasons: [{text: "", id: -1}],
      buttonIsLoading: false,
      explanationsToDelete: []
    }
  },
  methods: {
    updateSuggestion() {
      let suggestions = [];
      const commonTemplates = ["The [word|phrase] '____' appears in the text."];
      const nerAndRETemplates = [
        "The [word|phrase]  '____'  appears to the [right|left] of 'REPLACE' by at most  ____  words.",
        "The [word|phrase]  '____'  appears to the [right|left] of 'REPLACE' by at least  ____  words.",
        "The [word|phrase]  '____'  appears directly to the [right|left] of 'REPLACE'",
        "The [word|phrase]  '____'  appears within  ____  words of 'REPLACE'",
        "The [word|phrase]  '____'  appears within 1 word of 'REPLACE'",
      ]
      const reTemplates = [
        "The [word|phrase]  '____'  appears between 'REPLACE-1' and 'REPLACE-2'",
        "The word  '____'  is the only word between 'REPLACE-1' and 'REPLACE-2'",
        "There are no [more|less] than  ____   words between 'REPLACE-1' and 'REPLACE-2'",
        "There is one word between 'REPLACE-1' and 'REPLACE-2'",
        "'REPLACE-1' comes before 'REPLACE-2' by at [most|least]  ____  words",
        "'REPLACE-1' comes directly before 'REPLACE-2'",
      ]
      if (this.$store.getters.getProjectInfo.task === 3) {
        reTemplates.forEach(template => {
          const preFilledTemplate = template.replace("REPLACE-1", this.$store.getters["annotation/getRESelection"].sbj_text).replace("REPLACE-2", this.$store.getters["annotation/getRESelection"].obj_text);
          suggestions.push(preFilledTemplate);
        })
      }
      if (this.$store.getters.getProjectInfo.task > 1) {
        nerAndRETemplates.forEach(template => {
          let word = ""
          if (this.$store.getters.getProjectInfo.task === 2) {
            word = this.$store.getters["document/getCurDoc"].text.substring(this.$store.getters["annotation/getNERSelection"].selectionStart, this.$store.getters["annotation/getNERSelection"].selectionEnd)
          } else if (this.$store.getters.getProjectInfo.task === 3) {
            word = `[${this.$store.getters["annotation/getRESelection"].sbj_text} | ${this.$store.getters["annotation/getRESelection"].obj_text}]`
          }
          const preFilledTemplate = template.replace("REPLACE", word);
          suggestions.push(preFilledTemplate);
        })
      }
      commonTemplates.forEach(template => {
        suggestions.push(template);
      })
      return suggestions;
    },
    searchExplanationTemplate(queryString, cb) {
      let results = [];

      if (queryString) {
        results = [];
      } else {
        this.updateSuggestion().forEach(sug => {
          results.push({
            "value": sug,
            "link": sug
          })
        })
      }
      cb(results)
    },
    addAnotherExplanation() {
      this.reasons.push({text: "", id: -1})
    },
    deleteExplanation(index) {
      let targetExp = this.reasons[index];
      if (targetExp.id !== -1) {
        this.explanationsToDelete.push(targetExp.id);
      }
      this.reasons.splice(index, 1)
    },
    submitExplanations() {
      this.buttonIsLoading = true;

      const projectId = this.$store.getters.getProjectInfo.id;
      const documentId = this.$store.getters["document/getCurDoc"].id;
      const annotationId = this.$store.getters["explanation/getAnnotationInfo"].id;

      const promisesForDeletion = [];
      this.explanationsToDelete.forEach(explanationId => {
        let curPromise = this.$http.delete(`/projects/${projectId}/docs/${documentId}/annotations/${annotationId}/nl/${explanationId}`)
        promisesForDeletion.push(curPromise);
      })

      Promise.all(promisesForDeletion)
          .then(() => {
            const promisesToAdd = [];
            this.reasons.filter(reason => reason.id < 0).filter(reason => !!reason.text).forEach(reason => {
              let curPromise = this.$http.post(`/projects/${projectId}/docs/${documentId}/annotations/${annotationId}/nl/`, {text: reason.text})
              promisesToAdd.push(curPromise);
            })
            return Promise.all(promisesToAdd);
          })
          .then(() => {
            this.buttonIsLoading = false;
            this.$store.dispatch("document/fetchDocuments");
            this.$store.dispatch('explanation/hideExplanationPopup');
            this.$notify({type: "success", message: "Explanations updated", title: "Success"})
          })
          .catch(err => {
            console.error("failed to add explanations", err)
          });
    },
    dialogOpen() {
      if (this.$store.getters["explanation/getAnnotationInfo"]) {
        this.reasons = this.$store.getters["explanation/getAnnotationInfo"].explanations;
      }
      if (!this.reasons) {
        this.reasons = [{text: "", id: -1}]
      }
    }
  },
  computed: {
    dialogVisible: {
      get() {
        return this.$store.getters["explanation/getExplanationPopupInfo"].dialogVisible;
      },
      set() {
        this.$store.dispatch("explanation/hideExplanationPopup")
      }
    },
  },
  watch: {}
}
</script>

<style>
.explanation-dialog .el-dialog__body {
  padding-top: 0;
  padding-bottom: 0;
}
</style>
