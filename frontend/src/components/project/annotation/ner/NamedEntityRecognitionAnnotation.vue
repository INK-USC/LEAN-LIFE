<template>
  <el-card style="text-align: left; margin-top: 20px">
    <el-row>
      <el-tag>Text</el-tag>
    </el-row>
    <div style="line-height: 4; margin-top: 10px;" v-if="$store.getters['document/getCurDoc']"
         v-on:mouseup="setSelectedRange"
    >
      {{ this.$store.getters["document/getCurDoc"].text }}
    </div>


  </el-card>
</template>

<script>
export default {
  name: "NamedEntityRecognitionAnnotation",
  data() {
    return {
      selectionStart: -1,
      selectionEnd: -1,
    }
  },
  methods: {
    setSelectedRange() {
      let selectionStart = null;
      let selectionEnd = null;
      console.log("selection", window.getSelection(), document.getSelection())

      if (window.getSelection().anchorOffset !== window.getSelection().focusOffset) {
        const range = window.getSelection().getRangeAt(0);
        console.log("range", range);

        selectionStart = range.startOffset;
        selectionEnd = range.endOffset;

        const leadingWhiteSpace = range.commonAncestorContainer.data.search(/\S/);
        selectionStart -= leadingWhiteSpace;
        selectionEnd -= leadingWhiteSpace;

        const spanText = range.commonAncestorContainer.data.trim();
        const offSet = this.$store.getters["document/getCurDoc"].text.search(spanText);
        console.log("this.txt", this.$store.getters["document/getCurDoc"].text)
        console.log("span text", spanText);
        console.log("offset", offSet)

        selectionStart += offSet;
        selectionEnd += offSet;
      } else {
        console.log("reset selection")
      }

      // A selection has been made
      if (selectionStart !== null && selectionEnd != null) {
        // Trimming if Needed
        while (true) {
          if (!/[a-zA-Z0-9]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionStart, selectionStart + 1))) {
            selectionStart += 1;
          } else if (!/[a-zA-Z0-9]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionEnd - 1, selectionEnd))) {
            selectionEnd -= 1;
          } else {
            break;
          }
        }
        // Start of word not included
        while (selectionStart > 0 && /[a-zA-Z0-9-_]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionStart - 1, selectionStart))) {
          selectionStart -= 1;
        }
        // End of word not included
        while (selectionEnd < this.$store.getters["document/getCurDoc"].text.length && /[a-zA-Z0-9-]/.test(this.$store.getters["document/getCurDoc"].text.slice(selectionEnd, selectionEnd + 1))) {
          selectionEnd += 1;
        }
        console.log("selection start end", selectionStart, selectionEnd)
        this.$store.dispatch('annotation/setNERSelection', {
          "selectionStart": selectionStart,
          "selectionEnd": selectionEnd
        })
      }
    }
  },
  created() {
    // console.log(this.$store.getters["document/getCurDoc"])
  },
  computed: {}
}
</script>

<style scoped>

</style>
