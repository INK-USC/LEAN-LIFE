<template>
  <a-modal
    title="Natural Language Explanation 

    Please select a template below and:
      1) fill in the blanks
      2) decide between the two options in the '[]'

    You may also type out your own explanations, but please do so in a way similar to the provided templates."
    v-model="visible"
    @ok="handleOk"
    @cancel="handleCancel"
    width="85%"
  >
    <slot></slot>
    <div class="form-wrapper">
      <a-form>
        <a-form-item v-for="(entry, index) in explanations" :required="false">
          <a-auto-complete
            :dataSource="explanationSuggestions"
            class="explanation-auto-complete"
            style="width: 94%; margin-right: 8px;"
            @select="(value) => onExplanationChange(value, index)"
            @search=""
            @change="(value) => onExplanationChange(value, index)"
            :value=entry.text
            placeholder="Please Explain Why"
          />
          <!-- Display error message if nothing is provided -->
          <p v-if="errorMessage">{{ errorMessage }}</p>

          <a-icon
            v-if="explanations.length > 1"
            class="dynamic-delete-button"
            type="minus-circle-o"
            :disabled="explanations.length === 1"
            @click="() => removeExplanation(index)"
          />
        </a-form-item>
        <a-form-item v-if="explanations[explanations.length -1].text.length > 0">
          <a-button type="dashed" style="width: 94%" @click="addExplanation">
            <a-icon type="plus" /> Add Additional Explanation
          </a-button>
        </a-form-item>
      </a-form>
    </div>

    <a-button type="primary" @click="showLearnMoreDialog">Lean more</a-button>
    <a-modal v-model="learnMoreDialogVisible" title= "basic modal">
        <div style="display: flex; justify-content: space-between">
            <div style="width:50%">
                <h1>Token</h1>
                <a-list :data-source="tokens" style="overflow: auto; height: 400px;">
                    <a-list-item slot="renderItem" slot-scope="item, index">
                        <a-list-item-meta>
                            <div slot="title">
                                {{item}}
                            </div>
                        </a-list-item-meta>
                    </a-list-item>
                </a-list>
            </div>
            <div style="width: 50%">
                <h1>Grammars</h1>
                <a-list :data-source="grammars" style="overflow: auto; height: 400px;">
                    <a-list-item slot="renderItem" slot-scope="item, index">
                        <a-list-item-meta>
                            <div slot="title">
                                {{item}}
                            </div>
                        </a-list-item-meta>
                    </a-list-item>
                </a-list>
            </div>
        </div>
        <template slot="footer">
            <a-button @click="()=>learnMoreDialogVisible=false">Close</a-button>
        </template>

    </a-modal>
  </a-modal>
</template>
<script>

const NOT_YET_PUSHED_ID = -1;

/**
 * A modal to allow users to input natural language explanations for an
 * annotation. The modal itself only handles the input boxes, and it uses a Vue
 * slot to allow custom display of the original annotation.
 */
module.exports = {
  name: "nlModal",

  props: {
    lastAnnotationId: Number,
    docid: Number,
    selectedWords: Array,
    projectType: Number,
    currentExplanations: Array,
  },

  data() {
    return {
      selected: {},
      optionsShown: false,
      searchFilter: "",
      options: [],
      // Determines whether the ant design modal is visible. Should always be
      // true, since the parent nl-modal component is rendered with a v-if.
      visible: true,
      // Object holding the user's explanations. Key is index of the explanation
      // box and value is the explanation text.
      explanations: [],
      explanationSuggestions: [],
      errorMessage: "", //Display error message.
      idsToDelete: [],
      learnMoreDialogVisible: false,
      tokens: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
      grammars: [11,22,33,44,55,66,77,88,99,1010]
    };
  },
  created() {
    if (Object.keys(this.currentExplanations).length > 0) {
      this.explanations = this.currentExplanations;
    } else {
      this.explanations.push({"text" : "", "id": NOT_YET_PUSHED_ID});
    }
    
    this.resetSuggestions();
  },

  methods: {
    showLearnMoreDialog(){
        this.learnMoreDialogVisible= true;
    },
    resetSuggestions() {
      this.explanationSuggestions = [];
      
      const allProjectTemplates = [
        "The [word|phrase] '____' appears in the text."
      ];

      const nerAndRECondensedTemplates = [
        "The [word|phrase]  '____'  appears to the [right|left] of 'REPLACE' by at most  ____  words.",
        "The [word|phrase]  '____'  appears to the [right|left] of 'REPLACE' by at least  ____  words.",
        "The [word|phrase]  '____'  appears directly to the [right|left] of 'REPLACE'",
        "The [word|phrase]  '____'  appears within  ____  words of 'REPLACE'",
        "The [word|phrase]  '____'  appears within 1 word of 'REPLACE'",
      ];

      const reCondensedTemplates = [
        "The [word|phrase]  '____'  appears between 'REPLACE-1' and 'REPLACE-2'",
        "The word  '____'  is the only word between 'REPLACE-1' and 'REPLACE-2'",
        "There are no [more|less] than  ____   words between 'REPLACE-1' and 'REPLACE-2'",
        "There is one word between 'REPLACE-1' and 'REPLACE-2'",
        "'REPLACE-1' comes before 'REPLACE-2' by at [most|least]  ____  words",
        "'REPLACE-1' comes directly before 'REPLACE-2'",
      ];

      if (this.projectType == 3) {
        for (i in reCondensedTemplates) {
          const template = reCondensedTemplates[i];
          const partialFilledTemplate = template.replace("REPLACE-1", this.selectedWords[0]).replace("REPLACE-2", this.selectedWords[1]);
          this.explanationSuggestions.push(partialFilledTemplate);
        }
      }

      if (this.projectType > 1) {
        for (i in nerAndRECondensedTemplates) {
          const template = nerAndRECondensedTemplates[i];
          const word = this.selectedWords.length > 1 ? `[${this.selectedWords[0]}|${this.selectedWords[1]}]` : this.selectedWords[0];
          const partialFilledTemplate = template.replace("REPLACE", word);
          this.explanationSuggestions.push(partialFilledTemplate);
        }
      }

      for (i in allProjectTemplates) {
        const template = allProjectTemplates[i];
        this.explanationSuggestions.push(template);
      }
    },
    /**
     * Called whenever the user modifies one of the explanation boxes.
     * @param {string} value - The text of the explanation box changed.
     * @param {number} k - The index of the explanation box changed.
     */
    onExplanationChange(value, k) {

      if (this.explanations[k].text !== value) {
        this.explanations[k].text = value;
        if (this.explanations[k].id !== NOT_YET_PUSHED_ID) {
          this.idsToDelete.push(this.explanations[k].id);
          this.explanations[k].id = NOT_YET_PUSHED_ID;
        }
      }
     
      if (value.length > 2) {
          const fullAllProjectTemplates = [
            "The word '____' appears in the text.",
            "The phrase '____' appears in the text."
          ];

          const fullNERAndRESuggestions = [
          "The word  '____'  appears to the right of 'REPLACE' by at most  ____  words.",
          "The phrase  '____'  appears to the right of 'REPLACE' by at most  ____  words.",
          "The word  '____'  appears to the left of 'REPLACE' by at most  ____  words.",
          "The phrase  '____'  appears to the left of 'REPLACE' by at most  ____  words.",
          "The word  '____'  appears to the right of 'REPLACE' by at least  ____  words.",
          "The phrase  '____'  appears to the right of 'REPLACE' by at least  ____  words.",
          "The word  '____'  appears to the left of 'REPLACE' by at least  ____  words.",
          "The phrase  '____'  appears to the left of 'REPLACE' by at least  ____  words.",
          "The word  '____'  appears directly to the right of 'REPLACE'",
          "The phrase  '____'  appears directly to the right of 'REPLACE'",
          "The word  '____'  appears directly to the left of 'REPLACE'",
          "The phrase  '____'  appears directly to the left of 'REPLACE'",
          "The word  '____'  appears within  ____  words of 'REPLACE'",
          "The phrase  '____'  appears within  ____  words of 'REPLACE'",
          "The word  '____'  appears within 1 word of 'REPLACE'",
          "The phrase  '____'  appears within 1 word of 'REPLACE'",
        ];

        const fullRESuggestionss= [
          "The phrase  '____'  appears between 'REPLACE-1' and 'REPLACE-2'",
          "The word  '____'  appears between 'REPLACE-1' and 'REPLACE-2'",
          "The word  '____'  is the only word between 'REPLACE-1' and 'REPLACE-2'",
          "There are no more than  ____   words between 'REPLACE-1' and 'REPLACE-2'",
          "There are no less than  ____   words between 'REPLACE-1' and 'REPLACE-2'",
          "There is one word between 'REPLACE-1' and 'REPLACE-2'",
          "'REPLACE-1' comes before 'REPLACE-2' by at most  ____  words",
          "'REPLACE-1' comes before 'REPLACE-2' by at least  ____  words",
          "'REPLACE-1' comes directly before 'REPLACE-2'",
        ];

        const newSuggestions = []

        if (this.projectType == 3) {
          for (i in fullRESuggestionss) {
            const template = fullRESuggestionss[i];
            const partialFilledTemplate = template.replace("REPLACE-1", this.selectedWords[0]).replace("REPLACE-2", this.selectedWords[1]);
            newSuggestions.push(partialFilledTemplate);
          }
        }

        if (this.projectType > 1) {
          for (i in fullNERAndRESuggestions) {
            const template = fullNERAndRESuggestions[i];
            const word = this.selectedWords.length > 1 ? `[${this.selectedWords[0]}|${this.selectedWords[1]}]` : this.selectedWords[0];
            const partialFilledTemplate = template.replace("REPLACE", word);
            newSuggestions.push(partialFilledTemplate);
          }
        }

        for (i in fullAllProjectTemplates) {
          const template = fullAllProjectTemplates[i];
          newSuggestions.push(template);
        }

        const lowerCaseText = value.toLowerCase();
        const filteredSuggestions = [];
        if (lowerCaseText.startsWith("the word") && lowerCaseText.length < 50) {
          for (i = 0; i < newSuggestions.length; i++) {
            if (newSuggestions[i].toLowerCase().startsWith("the word")) {
              filteredSuggestions.push(newSuggestions[i]);
            }
          }
        } else if (lowerCaseText.startsWith("the phrase") && lowerCaseText.length < 50) {
          for (i = 0; i < newSuggestions.length; i++) {
            if (newSuggestions[i].toLowerCase().startsWith("the phrase")) {
              filteredSuggestions.push(newSuggestions[i]);
            }
          }
        } else {
          for (i = 0; i < newSuggestions.length; i++) {
            if (newSuggestions[i].toLowerCase().startsWith(lowerCaseText)) {
              filteredSuggestions.push(newSuggestions[i]);
            }
          }
        }
        this.explanationSuggestions = filteredSuggestions;
      } else if (value.length === 0) {
        if  (k < this.explanations.length-1) {
          this.explanations.splice(k, 1);
        }
        this.resetSuggestions();
      }
    },

    /**
     * Removes an explanation box.
     * @param {number} k - Index of the explanation box to remove.
     */
    removeExplanation(k) {
      if (this.explanations[k].id !== NOT_YET_PUSHED_ID) {
        this.idsToDelete.push(this.explanations[k].id);
      }
      this.explanations.splice(k, 1);
    },

    /**
     * Adds an explanation box.
     */
    addExplanation() {
      this.explanations.push({"text" : "", "id": NOT_YET_PUSHED_ID});
    },

    /**
     * Handles when the OK button on the modal is clicked.
     */
    handleOk() {
      if (this.explanations.length === 1 && this.explanations[0].text.trim().length === 0) {
        this.errorMessage = "Please provide at least one explanation.";
        return;
      }

      const promises = [];
      for (const id of this.idsToDelete) {
        promises.push(
          this.$http
            .delete(`docs/${this.docid}/annotations/${this.lastAnnotationId}/nl/${id}`)
            .catch((error) => {
              console.log(error);
        }));
      }

      Promise.all(promises).then(() => {
        const filteredExplanations = [];
        for (let i=0; i < this.explanations.length; i++) {
          const explanation = this.explanations[i];
          if (explanation.text.length > 0) {
            if (explanation.id === NOT_YET_PUSHED_ID) {
              const dbExplanation = {"text": explanation.text};
              this.$http
                .post(
                  `docs/${this.docid}/annotations/${this.lastAnnotationId}/nl/`,
                  dbExplanation
                )
                .then((response) => {
                  explanation.id = response.data.id;
                  this.explanations[i] = explanation;
                });
            }
            filteredExplanations.push(this.explanations[i])
          }
        }

        this.$emit('refresh-nl-explanations', filteredExplanations);
        this.hideModal();
      });
    },

    /**
     * Handles when the Cancel button on the modal is clicked.
     */
    handleCancel() {
      const deleteAnnotations = new Promise(() => {
        let i = this.explanations.length-1;
        while(i > -1) {
          if (this.explanations[i].id === NOT_YET_PUSHED_ID) {
            this.explanations.splice(i, 1)
          }
          i--;
        }
      });

      deleteAnnotations.then(this.hideModal(), this.hideModal());      
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

    selectOption(option) {
      this.selected = option;
      this.optionsShown = false;
      this.searchFilter = this.selected.name;
      this.$emit("selected", this.selected);
    },

    showOptions() {
      if (!this.disabled) {
        this.searchFilter = "";
        this.optionsShown = true;
      }
    },

    exit() {
      if (!this.selected.id) {
        this.selected = {};
        this.searchFilter = "";
      } else {
        this.searchFilter = this.selected.name;
      }
      this.$emit("selected", this.selected);
      this.optionsShown = false;
    },

    // Selecting when pressing Enter
    keyMonitor(event) {
      if (event.key === "Enter" && this.filteredOptions[0])
        this.selectOption(this.filteredOptions[0]);
    },
  },

  watch: {
    searchFilter() {
      if (this.filteredOptions.length === 0) {
        this.selected = {};
      } else {
        this.selected = this.filteredOptions[0];
      }
      this.$emit("filter", this.searchFilter);
    },
  },

  computed: {
    filteredOptions() {
      const filtered = [];
      const regOption = new RegExp(this.searchFilter, "ig");
      for (const option of this.options) {
        if (this.searchFilter.length < 1 || option.name.match(regOption)) {
          if (filtered.length < this.maxItem) filtered.push(option);
        }
      }
      return filtered;
    },
  },
};
</script>