<template>
  <div class="sentence-trigger-display-wrapper">
    <div class="tags has-addons">
      <span
        class="tag is-medium"
        v-bind:style="{
          color: labelColor,
          backgroundColor: labelBackgroundColor,
        }"
      >
        {{ labelText }}
      </span>
    </div>
    <div class="content sentence-display">
      <span class="text-sequence">{{ text }}</span>
    </div>
  </div>
</template>
<script>
/**
 * Displays the sentence / document along with a sentiment analysis label.
 */
module.exports = {
  name: 'sentenceLabelDisplay',

  props: {
    // The text color for the label.
    labelColor: String,
    // The background color for the label.
    labelBackgroundColor: String,
    // The name shown on the label.
    labelText: String,
    // The sentence text.
    text: String,
  },
};
</script>
