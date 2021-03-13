<template>
  <el-row style="margin-top: 20px">
    <div style="margin-bottom: 10px">
      <el-tag>Recommendation Section</el-tag>
    </div>

    <span v-for="chunk in recommendationChunks" :key="chunk.start_offset">
          <el-popover v-if="chunk.text_decoration">
            <el-col v-for="label of $store.getters['label/getLabels']" :key="label.id"
                    style="display: flex; width: fit-content">
              <span v-if="chunk.label.id === label.id" style="border: 1px solid red">
                <Label :labelInfo="label" :showShortcut="false"/>
              </span>
              <Label v-else :label-info="label" :showShortcut="false"/>
            </el-col>
            <span slot="reference"
                  :style="{textDecoration: chunk.text_decoration, cursor: 'pointer', marginRight: '5px'}"
                  @click="handleRecommendationClicked(chunk)">
              {{ curText.slice(chunk.start_offset, chunk.end_offset) }}
            </span>
          </el-popover>
          <span v-else>
            {{ curText.slice(chunk.start_offset, chunk.end_offset) }}
          </span>
        </span>
  </el-row>
</template>

<script>
//TODO incomplete. need to show label when clicked

import Label from "@/components/shared/Label";
import LabelListRow from "@/components/shared/LabelListRow";

export default {
  name: "Recommendation",
  components: {Label},
  data() {
    return {
      recommendations: null
    }
  },
  created() {
  },
  methods: {
    fetchRecommendation() {
      const projectId = this.$store.getters.getProjectInfo.id;
      const docId = this.$store.getters["document/getCurDoc"].id;
      const projectTask = this.$store.getters.getProjectInfo.task;

      this.$http.get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/recommendations/${this.$store.getters.getProjectInfo.task}/`)
          .then(res => {
            this.recommendations = res.recommendation;
            this.recommendations.sort((a, b) => a.start_offset - b.start_offset);
          })
    },
    handleRecommendationClicked(chunk) {
      console.log("recommendation clicked", chunk)
      this.$store.dispatch('annotation/setNERSelection', {
        "selectionStart": chunk.start_offset,
        "selectionEnd": chunk.end_offset
      })
    }
  },
  computed: {
    canFetch() {
      if (this.$store.getters.getProjectInfo && this.$store.getters["document/getCurDoc"]) {
        return true;
      } else {
        return false;
      }
    },
    recommendationChunks() {
      const chunks = [];
      if (!this.recommendations) {
        return [];
      }
      for (let i = 0; i < this.recommendations.length; i++) {
        const e = this.recommendations[i];
        e.text_decoration = "underline";
        if (i == 0) {
          if (e.start_offset !== 0) {
            chunks.push({
              text_decoration: "",
              start_offset: 0,
              end_offset: e.start_offset,
            });
          }
        } else {
          chunks.push({
            text_decoration: "",
            start_offset: this.recommendations[i - 1].end_offset,
            end_offset: e.start_offset,
          });
        }
        chunks.push(e);
      }
      const lastRec = this.recommendations[this.recommendations.length - 1];
      chunks.push({
        text_decoration: "",
        start_offset: lastRec.end_offset,
        end_offset: this.curText.length,
      });
      return chunks;
    },
    curText() {
      return this.$store.getters["document/getCurDoc"].text;
    }
  },
  watch: {
    canFetch: function (canFetchNow, canFetchBefore) {
      if (canFetchNow && !canFetchBefore) {
        this.fetchRecommendation()
      }
    }
  }

}
</script>

<style scoped>

</style>
