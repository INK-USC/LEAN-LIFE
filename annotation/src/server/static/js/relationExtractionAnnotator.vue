<template>
  <div style="padding-top: 15px;" class="re-annotator">
    <div
      v-for="c in chunks"
      :style="{ display: 'inline', position: 'relative', textAlign: 'left' }"
    >
      <span
        class="text-sequence no-highlight"
        v-bind:class="{
          tag: c.label,
        }"
        :style="getChunkStyle(c)"
        @click="handleChunkClick(c)"
        >{{ text.slice(c.start_offset, c.end_offset) }}</span
      >
      <span
        v-if="chunkIsSubject(c)"
        :style="{
          position: 'absolute',
          top: '-45px',
          left: '15px',
          right: '0px',
          pointerEvents: 'none',
          fontSize: '14px',
        }"
      >
        SUBJ
      </span>
      <span
        v-if="chunkIsObject(c)"
        :style="{
          position: 'absolute',
          top: '-45px',
          left: '15px',
          right: '0px',
          pointerEvents: 'none',
          fontSize: '14px',
        }"
      >
        OBJ
      </span>
    </div>
  </div>
</template>
<script>
module.exports = {
  name: 'relationExtractionAnnotator',

  props: {
    labels: Array, // [{id: Integer, color: String, text: String}]
    text: String,
    annotations: Array,
    // TODO: The recommendedLabels prop should be set.
    recommendedLabels: Array,
  },
  data() {
    return {
      // Jamin added:
      clickedChunks: [],
      nerPositions: [],
    };
  },

  methods: {
    clearClickedChunks() {
      this.clickedChunks = [];
      this.$emit('disable-labels');
    },

    chunkIsSubject(c) {
      return this.getChunkClickIndex(c) === 0;
    },

    chunkIsObject(c) {
      return this.getChunkClickIndex(c) === 1;
    },

    getChunkClickIndex(c) {
      for (let i = 0; i < this.clickedChunks.length; i++) {
        const { start_offset, end_offset } = this.clickedChunks[i];
        if (c.start_offset === start_offset && c.end_offset === end_offset) {
          return i;
        }
      }

      return -1;
    },

    chunkIsClicked(c) {
      return this.clickedChunks.some(
        ({ start_offset, end_offset }) => c.start_offset === start_offset && c.end_offset === end_offset,
      );
    },

    handleChunkClick(c) {
      if (this.chunkIsClicked(c)) {
        this.removeClickedChunk(c);
      } else {
        this.addClickedChunk(c);
      }
    },

    addClickedChunk(c) {
      if (this.clickedChunks.length == 2) {
        return;
      }

      this.clickedChunks.push(c);

      if (this.clickedChunks.length == 2) {
        const subject = this.clickedChunks[0];
        const object = this.clickedChunks[1];
        const subjectText = this.text.slice(subject.start_offset, subject.end_offset);
        const objectText = this.text.slice(object.start_offset, object.end_offset);
        subject.text = subjectText;
        object.text = objectText;
        this.$emit('enable-labels', [subject, object]);
      }
    },

    removeClickedChunk(c) {
      this.clickedChunks = this.clickedChunks.filter(
        ({ start_offset, end_offset }) => !(c.start_offset === start_offset && c.end_offset === end_offset),
      );

      if (this.clickedChunks.length < 2) {
        this.$emit('disable-labels');
      }
    },

    // TODO: As of right now the 'label' property in the chunk is just a boolean
    // so there aren't any styles attached to it.
    getChunkStyle(chunk) {
      const { label } = chunk;
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
      };
    },

    // addLabel(label) {
    //   if (this.clickedChunks.length < 2) {
    //     console.log("Less than 2 clicked entities, returning.");
    //     return;
    //   }

    //   const subject = this.clickedChunks[0];
    //   const object = this.clickedChunks[1];
    //   const relation = label;

    //   const subjectText = this.text.slice(subject.start_offset, subject.end_offset);
    //   const objectText = this.text.slice(object.start_offset, object.end_offset);
    //   const sbjFirst = (subject.start_offset < object.start_offset) ? true : false;

    //   this.$emit('add-label', {
    //     subject: { ...subject, text: subjectText },
    //     object: { ...object, text: objectText },
    //     relation,
    //     sbjFirst
    //   });
    // },

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

  watch: {
  },

  computed: {
    // TODO: Remove these default values and actually get the values from DB.
    sortedNERPositions() {
      this.nerPositions = [];
      for (let i = 0; i < this.annotations.length; i++) {
        const annotation = this.annotations[i];
        if (annotation["type"] === "ner") {
          this.nerPositions.push(annotation);
        }
      }
      this.nerPositions = this.nerPositions.sort(
        (a, b) => a.start_offset - b.start_offset
      );
      return this.nerPositions;
    },

    // TODO: Use actual label objects instead of IDs. Right now the 'label'
    // property of each chunk is just a boolean value, but if we want to be
    // able to display different colors for different types of named entities
    // or what not, we should pass in an object with the relavent information
    // instead.
    chunks() {
      const res = [];
      let left = 0;

      this.sortedNERPositions.forEach((mention) => {
        res.push({
          id: 0,
          label: null,
          start_offset: left,
          end_offset: mention.start_offset,
        });

        res.push({
          id: 0, // TODO: Use actual id.
          label: true, // TODO: Use actual label object or something, check backend.
          start_offset: mention.start_offset,
          end_offset: mention.end_offset,
        });

        left = mention.end_offset;
      });

      res.push({
        id: 0,
        label: null,
        start_offset: left,
        end_offset: this.text.length,
      });

      return res;
    },
  },
};
</script>
