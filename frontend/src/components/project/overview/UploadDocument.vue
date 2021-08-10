<template>
  <el-row>
    <el-col>
      <el-card>
        <div slot=header>
          <h3>Import your corpus below</h3>
        </div>
        <div>
          <div>In order to start the annotation process, a corpus <u>must</u> be uploaded</div>
          <div>We accept datasets in the following formats:</div>
          <ul>
            <li>
              <b>JSON (recommended)</b>
              <br/><u>Format:</u>
              <pre style="width: fit-content;padding-left: 20px; padding-right: 20px; height: fit-content">
                <code>
{
  "data" : [
    {<template v-if="this.$store.getters.getProjectInfo.task===1">
      "text" : "Louis Armstrong the great trumpet player lived in Corona.",
      "foo" : "bar",
      "bar" : "foo",
      "annotations": [
        {
          "label": "Label-Name-1"
    </template><template v-if="this.$store.getters.getProjectInfo.task===2">
      "text" : "Louis Armstrong the great trumpet player lived in Corona.",
      "foo" : "bar",
      "bar" : "foo",
      "annotations": [
        {
          "label": "Label-Name-1",
          "start_offset": 0,
          "end_offset": 5,
    </template><template v-if="this.$store.getters.getProjectInfo.task===3">
      "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
      "ner_annotations": [
          {
                "label": "", # can be empty string, or actual NER label
                "start_offset": 52,
                "end_offset": 58
          },
          {
                "label": "", # can be empty string, or actual NER label
                "start_offset": 0,
                "end_offset": 5
          }
          ...
      ],
      "metadata": {"foo" : "bar"},
      "annotations": [
        {
          "sbj_start_offset": 0,
          "sbj_end_offset": 5,
          "obj_start_offset": 52,
          "obj_end_offset": 58 
          </template><template v-if="this.$store.getters.getProjectInfo.explanation_type===2">"explanations": [
            "natural language explanation 1",
            "natural language explanation 2",
            ...
          ]
          </template><template v-if="this.$store.getters.getProjectInfo.explanation_type===3">"explanations": [
            [
              {
                "start_offset": 27,
                "end_offset": 41,
              },
              ...
            ]
            ...
          ]
      </template>}
      ]
    },
    ...
  ]
}
              </code>
							</pre>
              Each entry within <i>data</i> must have a key <i><b>text</b></i>. All other keys will be saved in a
              metadata
              dictionary associated
              with the text <br/>
              The <i><b>annotations</b></i> key in each entry is optional. And, can be used to populate the document 
              with label and explanations.
              <template v-if="this.$store.getters.getProjectInfo.task===3">
              <br/>The <i><b>ner_annotations</b></i> key in each entry provides entities in each document and is required to 
              </template>
            </li>
            <li v-if="this.$store.getters.getProjectInfo.task !== 3" style="margin-top: 40px">
              <b>CSV</b>--<u v-if="this.$store.getters.getProjectInfo.task !== 3">(Two formats are acceptable (but file
              must be using utf-8
              encoding):</u>
              <u v-else>BIO tagged data required</u>
              <ol style="">
                <li v-if="this.$store.getters.getProjectInfo.task !== 3">
                  With a header row, a column name must be <i><b>text</b></i>. All other columns will be saved in a
                  metadata dictionary
                  associated with the text
                </li>
                <u>Example 1:</u>
                <el-row>
                  <el-col :span="12">
                    <el-table :data="CSV_TABLE_EXAMPLE_1" stripe border>
                      <el-table-column prop="text" label="text"/>
                      <el-table-column prop="foo" label="foo" width="60"/>
                      <el-table-column prop="bar" label="bar" width="60"/>
                    </el-table>
                  </el-col>
                </el-row>


                <p v-if="this.$store.getters.getProjectInfo.task === 3">Needed Fields are: <b>word</b>, <b>label</b></p>
                <table style="" v-if="this.$store.getters.getProjectInfo.task === 3">
                  <tr>
                    <th>document_id</th>
                    <th>word</th>
                    <th>label</th>
                    <th>foo</th>
                    <th>bar</th>
                  </tr>
                  <tr>
                    <td>1</td>
                    <td>Louis</td>
                    <td>B-PER</td>
                    <td>bar</td>
                    <td>foo</td>
                  </tr>
                  <tr>
                    <td>1</td>
                    <td>Armstrong</td>
                    <td>I-PER</td>
                    <td>bar</td>
                    <td>foo</td>
                  </tr>
                  <tr>
                    <td>1</td>
                    <td>the</td>
                    <td>O</td>
                    <td>bar</td>
                    <td>foo</td>
                  </tr>
                </table>
                <li v-if="this.$store.getters.getProjectInfo.task !== 3" style="margin-top: 40px">No header, single
                  column file with just
                  text
                </li>
                <u v-if="this.$store.getters.getProjectInfo.task !== 3" style="margin-top: 10px">Example 2:</u>
                <el-row>
                  <el-col :span="12">
                    <el-table :data=" CSV_TABLE_EXAMPLE_1" stripe border :show-header="false">
                      <el-table-column prop="text"/>
                    </el-table>
                  </el-col>
                </el-row>
              </ol>
            </li>
            <li style="margin-top: 20px">
              <b>No commas can be in your text, which is why we strongly recommend using our json import process</b>
            </li>
          </ul>
        </div>

        <el-form :model="this.fileForm" style="text-align: center">
          <el-form-item>
            <el-radio v-model="fileForm.fileType" label="JSON" border>JSON file</el-radio>
            <el-radio v-model="fileForm.fileType" label="CSV" border v-if="this.$store.getters.getProjectInfo.task!=3">
              CSV file
            </el-radio>
          </el-form-item>
          <el-form-item label="">
            <el-upload :http-request="uploadFile" drag accept="text/json" ref="uploadInput" action="">
              <i class="el-icon-upload"></i>
              <div class="el-upload__text">Drop file here or <em>click to upload</em></div>
            </el-upload>
          </el-form-item>
        </el-form>
      </el-card>
    </el-col>
  </el-row>

</template>

<script>
import {ACTION_TYPE, CSV_TABLE_EXAMPLE_1, DIALOG_TYPE} from "@/utilities/constant";
// show user the correct format of their document to upload. and allow them to upload doc
export default {
  name: "UploadFile",
  data() {
    return {
      fileForm: {
        fileType: "JSON",
      },
      CSV_TABLE_EXAMPLE_1: CSV_TABLE_EXAMPLE_1
    }
  },
  methods: {
    uploadFile(param) {
      const formData = new FormData();
      const fileObj = param.file;
      formData.append("dataset", fileObj);
      formData.append("upload_type", (this.$store.getters.getProjectInfo.task == 1 || this.$store.getters.getProjectInfo.task == 2) ? 'pd' : 'ner');
      formData.append("format", this.fileForm.fileType.toLowerCase());
      this.$http
          .post(`/projects/${this.$store.getters.getProjectInfo.id}/docs/upload/`, formData, {
            headers: {
              ...this.$http.defaults.headers,
              "Content-Type": "multipart/form-data"
            }
          })
          .then(res => {
            console.log("upload succeed", res)
            this.$store.commit("updateActionRelatedInfo", {step: 2})
            this.$router.push({name: "LabelCreationSpace"})
          })
          .catch(err => {
            console.log(err)
          })
    },
  },
  created() {
    if (this.$store.getters.getActionType === ACTION_TYPE.CREATE) {
      this.$store.commit("showSimplePopup", DIALOG_TYPE.UploadDataSet);
    }
  }
}
</script>

<style scoped>
pre {
  background-color: rgb(245, 245, 245);
}
</style>
