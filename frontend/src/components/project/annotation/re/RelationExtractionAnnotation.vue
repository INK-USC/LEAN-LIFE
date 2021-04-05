<template>
  <el-card style="text-align: left; margin-top: 20px">
    <el-row>
      <el-tag>Text</el-tag>
    </el-row>

    <el-row style="line-height: 4">
      <div v-for="(chunk, index) in this.chunks" :key="index"
           :style="{ display: 'inline', position: 'relative', textAlign: 'left' }">
        <span :style="getChunkStyle(chunk)" @click="handleChunkClick(chunk)">
            {{ fullText.slice(chunk.start_offset, chunk.end_offset) }}
            <span v-if="chunkIsSubject(chunk)" :style="clickedChunkStyle">SUBJ</span>
            <span v-if="chunkIsObject(chunk)" :style="clickedChunkStyle">OBJ</span>
        </span>
      </div>
    </el-row>

    <el-row style="margin-top: 25px">
      <el-tag>Relations</el-tag>
    </el-row>
    <span v-if="this.$store.getters['document/getCurDoc']">
      <RelationDisplay
          v-for="(relation, index) in this.$store.getters['document/getCurDoc'].formattedAnnotations.filter(annotation=>annotation.type==='re')"
          :key="index" :relation="relation"/>
    </span>

  </el-card>
</template>

<script>
import RelationDisplay from "@/components/project/annotation/re/RelationDisplay";
import {ACTION_TYPE, DIALOG_TYPE} from "@/utilities/constant";
// annotation for RE
export default {
  name: "RelationExtractionAnnotation",
  components: {RelationDisplay},
  data() {
    return {
      clickedChunks: [],
      clickedChunkStyle: {
        position: 'absolute',
        top: '-45px',
        left: '15px',
        right: '0px',
        pointerEvents: 'none',
        fontSize: '14px',
      }
    }
  },

  methods: {
    // return the css style for the given chunk
    getChunkStyle(chunk) {
      const {label} = chunk;
      if (label == null) {
        return {};
      }

      if (this.chunkIsClicked(chunk)) {
        return {
          display: 'inline-block',
          borderColor: label.background_color,
          borderStyle: 'dashed',
          textDecoration: label.text_decoration,
          cursor: 'pointer',
          // color: label.text_color,
          // backgroundColor: label.background_color,
          lineHeight: 2,
          paddingLeft: "0.4em",
          paddingRight: "0.4em",
        };
      }
      return {
        // color: label.text_color,
        // backgroundColor: label.background_color,
        display: 'inline-block',
        borderColor: '#4A4A4A',
        borderStyle: 'dashed',
        borderWidth: '1px',
        textDecoration: label.text_decoration,
        cursor: 'pointer',
        '--chunk-hover-background-color': label.background_color,
        '--chunk-hover-text-color': label.text_color,
        lineHeight: 2,
        paddingLeft: "0.4em",
        paddingRight: "0.4em",

      };
    },
    // return if the annotated chunk is clicked
    chunkIsClicked(chunk) {
      return this.clickedChunks.some(
          ({start_offset, end_offset}) => chunk.start_offset === start_offset && chunk.end_offset === end_offset,
      );
    },
    // called when user clicked a chunk. if an annotated chunk is clicked, remove the annotation. else add annotation
    handleChunkClick(chunk) {
      if (this.chunkIsClicked(chunk)) {
        this.removeClickedChunk(chunk);
      } else {
        this.addClickedChunk(chunk);
      }
    },
    // add annotation to the chunk
    addClickedChunk(chunk) {
      if (this.clickedChunks.length == 2) {
        return;
      }
      this.clickedChunks.push(chunk);

      if (this.clickedChunks.length == 2) {
        const subject = this.clickedChunks[0];
        const object = this.clickedChunks[1];
        const subjectText = this.fullText.slice(subject.start_offset, subject.end_offset);
        const objectText = this.fullText.slice(object.start_offset, object.end_offset);
        subject.text = subjectText;
        object.text = objectText;
        //TODO enable labels
        this.$store.dispatch('annotation/setRESelection', {
          objStart: object.start_offset,
          objEnd: object.end_offset,
          sbjStart: subject.start_offset,
          sbjEnd: subject.end_offset,
          objText: object.text,
          sbjText: subject.text,
        })
      }
    },
    // remove annotation for the chunk
    removeClickedChunk(chunk) {
      this.clickedChunks = this.clickedChunks.filter(
          ({start_offset, end_offset}) => !(chunk.start_offset === start_offset && chunk.end_offset === end_offset)
      );
      if (this.clickedChunks.length < 2) {
        //TODO disable labels
        this.$store.dispatch('annotation/setRESelection', {
          objStart: -1,
          objEnd: -1,
          sbjStart: -1,
          sbjEnd: -1,
        })
      }
    },
    // check if the given chunk is a subject
    chunkIsSubject(chunk) {
      return this.getChunkClickIndex(chunk) === 0;
    },
    // check if the given chunk is a object
    chunkIsObject(chunk) {
      return this.getChunkClickIndex(chunk) === 1;
    },
    // get the index of the clicked chunk
    getChunkClickIndex(c) {
      for (let i = 0; i < this.clickedChunks.length; i++) {
        const {start_offset, end_offset} = this.clickedChunks[i];
        if (c.start_offset === start_offset && c.end_offset === end_offset) {
          return i;
        }
      }
      return -1;
    },


  },
  computed: {
    // get the full text of the current document
    fullText() {
      return this.$store.getters["document/getCurDoc"] ? this.$store.getters["document/getCurDoc"].text : "";
    },
    // generate sorted array of annotations by starting position
    sortedNERPositions() {
      let nerPositions = [];
      if (!this.$store.getters['document/getCurDoc']) {
        return [];
      }
      this.$store.getters['document/getCurDoc'].formattedAnnotations.filter(annotation => annotation['type'] === 'ner').forEach(annotation => {
        nerPositions.push(annotation);
      })

      nerPositions = nerPositions.sort((a, b) => a.start_offset - b.start_offset);
      return nerPositions;

    },
    // generate chunks to composite the full text
    chunks() {
      if (!this.$store.getters['document/getCurDoc']) {
        return [];
      }
      const res = [];
      let left = 0;
      this.sortedNERPositions.forEach((mention) => {
        res.push({id: 0, label: null, start_offset: left, end_offset: mention.start_offset});
        res.push({id: 0, label: true, start_offset: mention.start_offset, end_offset: mention.end_offset});
        left = mention.end_offset;
      })
      res.push({
        id: 0,
        label: null,
        start_offset: left,
        end_offset: this.$store.getters['document/getCurDoc'].text.length
      })
      return res;
    }
  },
  created() {
    if (this.$store.getters.getActionType === ACTION_TYPE.CREATE) {
      this.$store.commit("showAnnotationGuidePopup", DIALOG_TYPE.Annotation.RE);
    }
  }

}
</script>

<style scoped>

</style>
