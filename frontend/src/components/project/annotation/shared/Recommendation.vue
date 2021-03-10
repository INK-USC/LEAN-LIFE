<template>
  <el-row style="margin-top: 20px">
    <div style="margin-bottom: 10px">
      <el-tag>Recommendation Section</el-tag>
    </div>

    <span v-for="chunk in recommendationChunks" :key="chunk.start_offset">
      <el-popover v-if="chunk.text_decoration">
        <div>
          <span v-for="label in $store.getters['label/getLabels']" :key="label.id">
            {{ label.text }}
          </span>

        </div>
        <span slot="reference" :style="{textDecoration: chunk.text_decoration, cursor: 'pointer', marginRight: '5px'}">
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
export default {
  name: "Recommendation",
  data() {
    return {
      recommendations: null
    }
  },
  created() {
    this.fetchRecommendation();
  },
  methods: {
    fetchRecommendation() {
      this.$http.get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/${this.$store.getters["document/getCurDoc"].id}/recommendations/${this.$store.getters.getProjectInfo.task}/`)
          .then(res => {
            this.recommendations = res.recommendation;
            this.recommendations.sort((a, b) => a.start_offset - b.start_offset);
          })
    }
  },
  computed: {
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
}
</script>

<style scoped>

</style>
