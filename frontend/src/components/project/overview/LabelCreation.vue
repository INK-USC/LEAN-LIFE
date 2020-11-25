<template>
  <el-row>
    <el-card>
      <div slot="header"><h3>Label Space Creation</h3></div>
      <div style="text-align: left">
        <el-row>
          <el-col :span="6">
            <el-tooltip content="Available labels for annotators to pick from when annotating." placement="top">
              <b>Current Label Space</b>
            </el-tooltip>

          </el-col>
          <el-col :span="18" style="display: flex; flex-wrap: wrap">
            <Label v-for="label of existingLabels" :key="label.id" :labelInfo="label" style="margin-right: 10px"
                   @deleted="fetchLabels"/>
          </el-col>
        </el-row>

        <el-divider/>

        <el-form ref="labelCreationForm" :model="labelCreationForm">
          <el-row class="row">
            <el-col :span="6">
              <b>Label Preview</b>
            </el-col>
            <el-col :span="18">
              <Label :labelInfo="submissionForm"/>
            </el-col>
          </el-row>
          <el-row class="row" style="margin-top: 20px;">
            <el-col :span="6">
              <b>Label Name</b>
            </el-col>
            <el-col :span="18">
              <el-input v-model="labelCreationForm.text" placeholder="Labels must be named to be saved"></el-input>
            </el-col>
          </el-row>
          <el-row style="margin-top: 20px;">
            <el-col :span="6">
              <b>Shortcut Key</b>
            </el-col>
            <el-col :span="18">
              <el-select v-model="labelCreationForm.shortcutKey">
                <el-option v-for="option in shortcutOptions" :key="option" :label="option"
                           :value="option.length>1? '':option" :disabled="option.length>1? true: false"/>
              </el-select>
              <el-checkbox v-model="labelCreationForm.hasCtrl" border size="large" style="margin-left: 40px">C: Ctrl
              </el-checkbox>
              <el-checkbox v-model="labelCreationForm.hasShift" border size="large">S: Shift</el-checkbox>
            </el-col>
          </el-row>
          <el-row style="margin-top: 20px;">
            <el-col :span="6">
              <b>Background Color</b>
            </el-col>
            <el-col :span="18">
              <el-color-picker v-model="labelCreationForm.background_color"/>
            </el-col>
          </el-row>
          <el-row style="margin-top: 20px;">
            <el-col :span="6">
              <b>Text Color</b>
            </el-col>
            <el-col :span="18">
              <el-color-picker v-model="labelCreationForm.text_color"/>
            </el-col>
          </el-row>
          <el-row>
            <el-button type="primary" :disabled="!submissionForm.text" @click="createLabel">Save Label</el-button>
            <el-button type="danger" :disabled="!formHasChanged" @click="resetLabel">Reset Current Label</el-button>
            <el-button type="success" :disabled="!existingLabels ||existingLabels.length==0" @click="goNextStep">Done
            </el-button>
          </el-row>
        </el-form>
      </div>
    </el-card>

  </el-row>

</template>

<script>
import Label from "@/components/shared/Label";

const DEFAULT_FORM = {
  text: "",
  background_color: '#209cee',
  text_color: '#ffffff',
  shortcutKey: "",
  hasCtrl: false,
  hasShift: false,
};

export default {
  name: "LabelCreation",
  components: {Label},
  data() {
    return {
      existingLabels: [],
      labelCreationForm: {},
      shortcutOptions: ["Please select one", ..."abcdefghijklmnopqrstuvwxyz".split('')]
    }
  },
  methods: {
    fetchLabels() {
      this.$http.get(`/projects/${this.$store.getters.getProjectInfo.id}/labels/`).then(res => {
        console.log(res)
        this.existingLabels = res;
      })
    },
    createLabel() {
      this.$http.post(`/projects/${this.$store.getters.getProjectInfo.id}/labels/`, this.submissionForm).then(res => {
        console.log("create label", res)
        this.resetLabel();
        this.fetchLabels();
      })
    },
    resetLabel() {
      this.labelCreationForm = JSON.parse(JSON.stringify(DEFAULT_FORM))
    },
    goNextStep() {

    }
  },
  created() {
    this.labelCreationForm = JSON.parse(JSON.stringify(DEFAULT_FORM))
    this.fetchLabels()
  },
  computed: {
    submissionForm() {
      let shortcut = this.labelCreationForm.hasCtrl ? "ctrl" : "";
      shortcut += this.labelCreationForm.hasShift ? "shift" : "";
      shortcut += this.labelCreationForm.shortcutKey ? this.labelCreationForm.shortcutKey : "";
      shortcut = shortcut.split("").join("-");

      let res = {
        text: this.labelCreationForm.text,
        shortcut: shortcut,
        background_color: this.labelCreationForm.background_color,
        text_color: this.labelCreationForm.text_color,
      }
      return res;
    },
    formHasChanged() {
      return this.labelCreationForm === {
        text: "",
        background_color: '#209cee',
        text_color: '#ffffff',
        shortcutKey: "",
        hasCtrl: false,
        hasShift: false,
      }
    }
  },
  watch: {}
}
</script>

<style scoped>
.row {;
}
</style>
