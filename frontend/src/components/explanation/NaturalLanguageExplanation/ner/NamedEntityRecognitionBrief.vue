<template>
  <el-row>
        <span v-for="(chunk, index) in this.chunks" :key="index"
              :style="{color: id2label[chunk.label] ? id2label[chunk.label].text_color: '',
              backgroundColor: id2label[chunk.label] ? id2label[chunk.label].background_color : '',
              padding: '10px 10px 10px 10px', fontSize: '18px',
              borderRadius: id2label[chunk.label]? (id2label[chunk.label].text_color? '8px': 0) : 0}"
        >
          {{ fullText.slice(chunk.start_offset, chunk.end_offset) }}
        </span></el-row>
</template>

<script>
export default {
  name: "NamedEntityRecognitionBrief",
  methods: {
    makeLabel(startOffset, endOffset) {
      const label = {
        id: 0,
        label: -1,
        start_offset: startOffset,
        end_offset: endOffset,
      };
      return label;
    },
  },
  computed: {
    chunks() {
      if (!this.$store.getters["document/getCurDoc"]) {
        return [];
      }
      let sortedEntityPositions = this.$store.getters["document/getCurDoc"] ? this.$store.getters["document/getCurDoc"].formattedAnnotations : [];

      const res = [];
      let left = 0;
      for (let i = 0; i < sortedEntityPositions.length; i++) {
        const e = sortedEntityPositions[i];
        const l = this.makeLabel(left, e.start_offset);
        res.push(l);
        res.push(e);
        left = e.end_offset;
      }
      const l = this.makeLabel(left, this.$store.getters["document/getCurDoc"].text.length);
      res.push(l);
      return res
    },
    fullText() {
      return this.$store.getters["document/getCurDoc"].text;
    },
    id2label() {
      const id2label = {};
      // default value;
      id2label[-1] = {
        text_color: "",
        background_color: "",
        text_decoration: "",
      };

      for (let i = 0; i < this.$store.getters['label/getLabels'].length; i++) {
        const label = this.$store.getters["label/getLabels"][i];
        label.text_decoration = "";
        id2label[label.id] = label;
      }
      return id2label;
    },
  }
}
</script>

<style scoped>

</style>
