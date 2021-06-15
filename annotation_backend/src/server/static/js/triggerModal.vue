<template>
  <a-modal title="Explanation" v-model="visible" @ok="handleOk" @cancel="handleCancel" width="60%">
    <slot></slot>
    <div v-for="(labels, labelId) in triggers" class="trigger-wrapper">
      <div class="trigger-header">
        <p>
          <b>Trigger {{ labelId + 1 }}:</b>
        </p>
        <a-icon v-if="triggers.length > 1" class="trigger-delete-button" type="close-square"
          :disabled="triggers.length === 1" @click="() => removeTriggerInput(labelId)" />
      </div>
      <div class="content sentence-display" v-on:mouseup="setSelectedRange(labelId)">
        <span class="text-sequence" v-for="chunk in chunks" v-bind:class="getChunkClass(chunk, labelId)" v-bind:style="getChunkStyle(chunk, labelId)">
          {{ text.slice(chunk.start_offset, chunk.end_offset)}}
          <button class="delete is-small" v-if="chunk.labelId === labelId" @click="removeTrigger(chunk)"/>
        </span>
      </div>
    </div>
    <a-button type="dashed" style="width: 90%" :disabled="triggers.length >= triggerLabels.length"
      @click="addTriggerInput">
      <a-icon type="plus" /> Add Additional Trigger
    </a-button>
    <!-- Display error message if nothing is provided -->
    <p v-if="errorMessage">{{ errorMessage }}</p>
  </a-modal>
</template>
<script>

// Const def to represent an unlabeled chunk.
const UNLABELED_LABEL_ID = -1;
// Const def to represent a chunk that is the original selected text.
const SELECTION_LABEL_ID = -2;
// Const def to represent a chunk that is the original selected text.
const NOT_YET_PUSHED_ID = -3;

/**
 * A modal to allow users to highlight trigger word explanations for an
 * annotation. The modal itself only handles the trigger selection, and it uses
 * a Vue slot to allow custom display of the original annotation.
 */
module.exports = {
  name: "triggerModal",

  props: {
    lastAnnotationId: Number,
    docid: Number,
    triggerLabels: Array, // [{id: Integer, color: String, text: String}]
    text: String,
    // The original selection(s) that the user made. These selections will be
    // disabled when selecting triggers.
    originalSelections: {
      type: Array,
      default: () => [],
    },
    currentTriggers: Array,
  },

  data() {
    return {
      startOffset: 0,
      endOffset: 0,
      visible: true,
      // Triggers is an array of arrays, where the top-level index is the label
      // ID. Each nested array holds the labels corresponding to that specific
      // label ID. Each label is represented as an object with the following
      // properties: { start_offset, end_offset, labelId }
      triggers: [[]], // Show one default trigger. 
      errorMessage: "", //Display error message.
      idsToDelete: [],

    };
  },

  methods: {
    /**
     * Returns the HTML classes that we should apply to a chunk.
     */
    getChunkClass(chunk, labelId) {
      return {
        tag: chunk.labelId === labelId,
        // Prevent the chunk from being selected if it's highlighted in another
        // trigger, or it was the original selection.
        noselect: 
          (chunk.labelId !== UNLABELED_LABEL_ID && chunk.labelId !== labelId) || chunk.labelId === SELECTION_LABEL_ID,
      };
    },

    /**
     * Returns the CSS style to apply to a chunk.
     */
    getChunkStyle(chunk, labelId) {
      // Check if the chunk matches the current label ID. If so, we want to
      // highlight the text span.
      if (chunk.labelId === labelId) {
        return {
          color: this.id2label[labelId].text_color,
          backgroundColor: this.id2label[labelId].background_color,
          textDecoration: this.id2label[labelId].text_decoration,
        };
      }
      // If the chunk is a label but doesn't match our current label ID, we want
      // to mark it as grey to discourage users from selecting again.
      if (chunk.labelId !== UNLABELED_LABEL_ID) {
        return {
          color: "#bfbfbf",
        };
      }
      // Return empty style for a chunk that isn't a label.
      return {};
    },

    /**
     * Adds a new trigger input field.
     */
    addTriggerInput() {
      if (this.triggers[this.triggers.length - 1].length > 0) {
        this.triggers.push([]);
      }
    },

    /**
     * Removes the target trigger input field and reassigns label IDs so that
     * they are consecutive.
     */
    removeTriggerInput(removeLabelId) {
      const labels = this.triggers[removeLabelId];
      for (const l of labels) {
        if (l.pk_id !== NOT_YET_PUSHED_ID) {
          this.idsToDelete.push(l.pk_id);
        }
      }
      this.triggers.splice(removeLabelId, 1);
      for (let newLabelId = 0; newLabelId < this.triggers.length; newLabelId++) {
        for (const l of this.triggers[newLabelId]) {
          if (l.labelId !== newLabelId) {
            this.idsToDelete.push(l.pk_id);
            l.labelId = newLabelId;
            l.pk_id = NOT_YET_PUSHED_ID;
          }
        }
      }
    },

    /**
     * Handles when the OK button on the modal is clicked.
     */
    handleOk() {
      //If there is no trigger, display the error message and do nothing
      if (this.triggers[0].length === 0) {
        this.errorMessage = "Please provide at least one trigger.";
        return;
      }

      const promises = [];
      for (const id of this.idsToDelete) {
        promises.push(
          this.$http
            .delete(`docs/${this.docid}/annotations/${this.lastAnnotationId}/trigger/${id}`)
            .catch((error) => {
              console.log(error);
        }));
      }

      Promise.all(promises).then(() => {
        const docVersionOfTriggers = [[], [], [], []];
        for (const lt of this.triggers) {
          for (const l of lt) {
            const triggerAnnotationPost = {
              start_offset: l.start_offset,
              end_offset: l.end_offset,
              trigger_id: l.labelId,
            };
            if (l.pk_id === NOT_YET_PUSHED_ID) {
              this.$http
              .post(
                `docs/${this.docid}/annotations/${this.lastAnnotationId}/trigger/`,
                triggerAnnotationPost
              )
              .then((response) => {
                const docSubTrigger = {
                  id: response.data.id,
                  start_offset: l.start_offset,
                  end_offset: l.end_offset,
                  trigger_id: l.labelId,
                  base_ann_id: l.base_ann_id,
                };
                docVersionOfTriggers[l.labelId].push(docSubTrigger);
              });
            } else {
              const docSubTrigger = {
                  id: l.pk_id,
                  start_offset: l.start_offset,
                  end_offset: l.end_offset,
                  trigger_id: l.labelId,
                  base_ann_id: l.base_ann_id,
                };
                docVersionOfTriggers[l.labelId].push(docSubTrigger);
            } 
          }
        }
        this.$emit('refresh-triggers', docVersionOfTriggers);
        this.hideModal();
      });
    },

    /**
     * Handles when the Cancel button on the modal is clicked.
     */
    handleCancel() {
      this.hideModal();
    },

    /**
     * Hides the modal. This is pretty jank b/c the parent component is rendered
     * via a v-if, so we always want to keep the modal "visible," and just emit
     * the 'close' event so the v-if hides the modal.
     */
    hideModal() {
      this.visible = true;
      this.$emit('close');
    },

    setSelectedRange(labelId) {
      let selectionStart = null;
      let selectionEnd = null;
      if (window.getSelection().anchorOffset !== window.getSelection().focusOffset) {
        const range = window.getSelection().getRangeAt(0);
        selectionStart = range.startOffset;
        selectionEnd = range.endOffset;

        // const leadingWhiteSpace = range.commonAncestorContainer.data.search(/\S/);
        // selectionStart = selectionStart - leadingWhiteSpace;
        // selectionEnd = selectionEnd - leadingWhiteSpace;

        const spanText = range.commonAncestorContainer.data.trim();
        const offSet = this.text.search(spanText);

        selectionStart = selectionStart + offSet;
        selectionEnd = selectionEnd + offSet;
      }

      if (selectionStart !== null && selectionEnd !== null) {
        this.startOffset = selectionStart;
        this.endOffset = selectionEnd;
        // Trimming if Needed
        while (true) {
          if (!/[a-zA-Z0-9]/.test(this.text.slice(this.startOffset, this.startOffset+1))) {
            this.startOffset += 1;
          } else if (!/[a-zA-Z0-9]/.test(this.text.slice(this.endOffset-1, this.endOffset))) {
            this.endOffset -= 1;
          } else {
            break;
          }
        }

        // Start of word not included
        while (this.startOffset > 0 && /[a-zA-Z0-9-_]/.test(this.text.slice(this.startOffset-1, this.startOffset))) {
          this.startOffset -= 1;
        }

        // End of word not included
        while (this.endOffset < this.text.length && /[a-zA-Z0-9-]/.test(this.text.slice(this.endOffset, this.endOffset+1))) {
          this.endOffset += 1;
        }

        this.addTrigger(labelId);
      }
    },

    /**
     * Returns true if the given text selection is valid trigger. This function
     * performs some basic checks to distinguish clicks from highlights, as well
     * as ensuring previously-highlighted phrases are not selected again.
     */
    validRange() {
      // Check that the user made a text selection and not a click.
      if (this.startOffset === this.endOffset) {
        return false;
      }
      // Check that the selection length is not longer then the actual text
      // length.
      if (this.startOffset > this.text.length || this.endOffset > this.text.length) {
        return false;
      }
      // Check that the start/end offsets are greater than 0.
      if (this.startOffset < 0 || this.endOffset < 0) {
        return false;
      }
      // Check that the user actually selected text and not whitespace.
      if (this.text.slice(this.startOffset, this.endOffset).trim() === "") {
        return false;
      }

      // Check that the text selection does not overlap with any previous
      // triggers.
      for (const labels of this.triggers) {
        for (const l of labels) {
          if (
            this.startOffset <= l.end_offset &&
            l.start_offset <= this.endOffset
          ) {
            return false;
          }
        }
      }

      return true;
    },

    /**
     * Adds a trigger for the given label ID.
     */
    addTrigger(labelId) {
      if (this.validRange()) {
        const triggerAnnotationUI = {
          pk_id: NOT_YET_PUSHED_ID,
          start_offset: this.startOffset,
          end_offset: this.endOffset,
          labelId,
          base_ann_id: this.lastAnnotationId,
        };
        this.triggers[labelId].push(JSON.parse(JSON.stringify(triggerAnnotationUI)));
      }
    },

    /**
     * Removes the trigger associated with the following chunk.
     */
    removeTrigger(chunk) {
      const subTriggers = this.triggers[chunk.labelId];
      // Try to find the trigger associated with the following chunk.
      const removeIdx = subTriggers.findIndex(
        (l) =>
          l.start_offset === chunk.start_offset &&
          l.end_offset === chunk.end_offset
      );
      if (removeIdx > -1) {
        subTriggers.splice(removeIdx, 1);
      } else {
        throw new Error(
          "Could not find the chunk to remove, this is most likely a bug!"
        );
      }
      if (chunk.pk_id !== NOT_YET_PUSHED_ID) {
        this.idsToDelete.push(chunk.pk_id);
      }
    },

    makeLabel(startOffset, endOffset, label) {
      return {
        label,
        start_offset: startOffset,
        end_offset: endOffset,
      };
    },
  },

  created() {
    for (let i=0; i < this.currentTriggers.length; i++) {
      if (this.currentTriggers[i].length > 0) {
        if (i === this.triggers.length) {
          this.triggers.push([]);
        }
        for (let j=0; j <this.currentTriggers[i].length; j++) {
          currSubTrigger = this.currentTriggers[i][j];
          const subTriggerUI = {
            pk_id: currSubTrigger.id,
            start_offset: currSubTrigger.start_offset,
            end_offset: currSubTrigger.end_offset,
            labelId: currSubTrigger.trigger_id,
            base_ann_id: currSubTrigger.base_ann_id,
          }
          this.triggers[i].push(subTriggerUI);
        }
      }
    }
  },

  computed: {
    id2label() {
      const id2label = {};
      // default value;
      id2label[-1] = {
        text_color: "",
        background_color: "",
        text_decoration: "",
        shortcut: "",
      };

      for (let i = 0; i < this.triggerLabels.length; i++) {
        const label = this.triggerLabels[i];
        label.text_decoration = "";
        id2label[label.id] = label;
      }
      return id2label;
    },

    /**
     * Returns an array of chunks after considering ALL labels as well as the
     * original selection span.
     */
    chunks() {
      const res = [];
      let left = 0;
      const allTriggers = this.triggers.flat();
      // Add a label for the original text selection as well.
      for (const l of this.originalSelections) {
        allTriggers.push({
          pk_id: SELECTION_LABEL_ID,
          start_offset: l.start_offset,
          end_offset: l.end_offset,
          labelId: SELECTION_LABEL_ID,
        });
      }

      // Sort the labels from start to end.
      allTriggers.sort((a, b) => a.start_offset - b.start_offset);

      // console.log("allTriggers = ", allTriggers);

      // Loop through the text and create all chunks.
      for (const l of allTriggers) {
        res.push({
          pk_id: UNLABELED_LABEL_ID,
          start_offset: left,
          end_offset: l.start_offset,
          labelId: UNLABELED_LABEL_ID,
        });

        res.push({
          pk_id: l.pk_id,
          start_offset: l.start_offset,
          end_offset: l.end_offset,
          labelId: l.labelId,
        });

        left = l.end_offset;
      }

      res.push({
        pk_id: UNLABELED_LABEL_ID,
        start_offset: left,
        end_offset: this.text.length,
        labelId: UNLABELED_LABEL_ID,
      });

      return res;
    },
  },
};
</script>
