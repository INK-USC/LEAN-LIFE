<template>
  <div class="content sentence-display">
    <span
      class="text-sequence"
      v-for="c in chunks"
      v-if="id2Label[c.label]"
      v-bind:class="{ tag: id2Label[c.label].text_color }"
      v-bind:style="{
        color: id2Label[c.label].text_color,
        backgroundColor: id2Label[c.label].background_color,
        textDecoration: id2Label[c.label].text_decoration,
      }"
      >{{ text.slice(c.start_offset, c.end_offset) }}</span
    >
  </div>
</template>
<script>
/**
 * Displays the sentence / document along with the specific text selections
 * highlighted.
 */
module.exports = {
  name: 'sentenceHighlightDisplay',

  props: {
    labelPositions: Array,
    labels: Array,
    text: String,
  },

  computed: {
    chunks() {
      const res = [];
      let left = 0;
      this.labelPositions
        .sort((a, b) => a.start_offset - b.start_offset)
        .forEach((l) => {
          if (l.start_offset > 0) {
            res.push({
              start_offset: left,
              end_offset: l.start_offset,
              label: -1,
            });
          }
          res.push(l);

          left = l.end_offset;
        });

      if (left < this.text.length) {
        res.push({
          start_offset: left,
          end_offset: this.text.length,
          label: -1,
        });
      }
      return res;
    },

    id2Label() {
      const id2Label = {};
      // default value;
      id2Label[-1] = {
        text_color: '',
        background_color: '',
        text_decoration: '',
        shortcut: '',
      };

      this.labels.forEach((l) => {
        l.text_decoration = '';
        id2Label[l.id] = l;
      });

      return id2Label;
    },
  },
};
</script>
