<template>
  <div>
    <el-dialog :visible.sync="dialogVisible" width="50%" style="text-align: center;">
      <h1 slot="title">
        <i class="el-icon-info"></i>
        {{ this.dialogContents[this.$store.getters.getSimplePopupInfo.targetDialogType].title }}
      </h1>
      <div v-for="msg in this.dialogContents[this.$store.getters.getSimplePopupInfo.targetDialogType].messages"
           :key="msg" class="message">
        {{ msg }}
      </div>
      <span slot="footer">
				<el-button type="primary" @click="()=>this.$store.commit('hideSimplePopup')">OK</el-button>
			</span>
    </el-dialog>
  </div>

</template>

<script>
import {DIALOG_TYPE} from "@/utilities/constant";
// tell user which step they are on. only show when they first create the project. step is be indicated by DIALOG_TYPE.
export default {
  name: "SimplePopup",
  data() {
    return {
      dialogContents: {
        "": {title: "", messages: []}
      }
    }
  },
  created() {
    this.dialogContents[DIALOG_TYPE.UploadDataSet] = {
      title: "Upload Dataset",
      messages: ["Your Project has been created!",
        "You can always go back and edit your project setup on the Project Page.",
        "You must now upload a dataset for your project.",
        "You can later upload more documents.",
        "They will be appended into existing documents "
      ]
    };
    this.dialogContents[DIALOG_TYPE.CreatingLabels] = {
      title: "Creating Labels",
      messages: ["Great! Now annotators have documents to annotate!",
        "Now its time to create the set of possible labels.",
        "Annotators may use them to annotate your uploaded documents.",
        "You must create at least one label before navigating away from this page",
        "We highly recommend giving your labels a shortcut key, so that annotators can annotate quickly.",
        "To save a created label, please hit 'Save Label'.",
        "When you are finished creating your label space, please hit 'Done'"
      ]
    };
    this.dialogContents[DIALOG_TYPE.ConfiguringOptionalAnnotationSettings] = {
      title: "Configuring Optional Annotation Settings",
      messages: ["Great! Now annotators have documents to annotate and labels to apply!",
        "There are some remaining optional settings you can set here.",
        "As you are the project creator, what you decide here will be the default for all annotators, though annotators may override if they wish to.",
        "Hover over the question marks to understand what each setting means!",
        "When you are finished, hit the save button to save your selections.",
      ]
    }
  },
  computed: {
    dialogVisible: {
      get() {
        return this.$store.getters.getSimplePopupInfo.dialogVisible
      },
      set() {
        this.$store.commit("hideSimplePopup")
      }
    }
  }

}
</script>

<style scoped>
.message {
  font-size: 20px;
}
</style>
