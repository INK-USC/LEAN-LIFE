<template>
  <div>
    <h1>Export Annotations</h1>
    <h4>Create and Export a dataset with current Annotations</h4>
    <div>
      <p>The following download options are possible:</p>
      <ul style="list-style-type:disc; list-style-position:outside;">
        <li>
          <b>JSON(Recommended)</b>
          <pre style="display: flex">
              <code v-if="this.$store.getters.getProjectInfo.task === 1">
{
  "data": [
      {
            "doc_id": 1,
            "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
            "annotations": [
                {
                      "annotation_id": 18,
                      "label": "postive",
                      "explanation": [
                          "The word 'great' appears in the sentence."
                      ]
                }
                ...
            ],
            "user": 4,
            "metadata": {"foo" : "bar"}
      }
      ...
  ]
}
              </code>
              <code v-if="this.$store.getters.getProjectInfo.task === 2">
{
  "data": [
      {
            "doc_id": 1,
            "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
            "annotations": [
                {
                      "annotation_id": 18,
                      "label": "LOC",
                      "start_offset": 52,
                      "end_offset": 58,
                      "explanation": [
                          "lived in"
                      ]
                }
                ...
            ],
            "user": 4,
            "metadata": {"foo" : "bar"}
      }
      ...
  ]
}
              </code>
              <code v-if="this.$store.getters.getProjectInfo.task === 3">
{
     "data": [
          {
               "doc_id": 1,
               "text": "Louis Armstrong, the great trumpet player, lived in Corona.",
               "annotations": [
                    {
                         "annotation_id": 18,
                         "label": "per:occupation",
                         "sbj_start_offset": 0,
                         "sbj_end_offset": 15,
                         "obj_start_offset": 27,
                         "obj_end_offset": 41,
                         "explanation": [
                              "The token  ','  appears between 'Louis Armstrong' and 'trumpet player'",
                              "The token ',' appears to the right of 'trumpet player' by no more than 2 words",
                              "There are no more than 5 words between 'Louis Armstrong' and 'trumpet player'"
                         ]
                    }
                    ...
               ],
               "user": 4,
               "metadata": {"foo" : "bar"}
          }
          ...
     ]
}
              </code>
            </pre>
        </li>
        <li v-if="this.$store.getters.getProjectInfo.task !== 1">
          <b>CSV</b>
          <p></p>
          <table style="width:100%" v-if="this.$store.getters.getProjectInfo.task === 2">
            <tr>
              <th>document_id</th>
              <th>word</th>
              <th>label</th>
              <th>metadata</th>
              <th>explanation</th>
            </tr>
            <tr>
              <td>1</td>
              <td>Louis</td>
              <td>B-PER</td>
              <td>{}</td>
              <td>, the great trumpet player,:*:*:lived in</td>
            </tr>
            <tr>
              <td>1</td>
              <td>Armstrong</td>
              <td>I-PER</td>
              <td></td>
              <td></td>
            </tr>
            <tr>
              <td>1</td>
              <td>,</td>
              <td>O</td>
              <td></td>
              <td></td>
            </tr>
          </table>
          <table style="width:100%" v-if="this.$store.getters.getProjectInfo.task === 3">
            <tr>
              <th>document_id</th>
              <th>entity_1</th>
              <th>entity_2</th>
              <th>label</th>
              <th>metadata</th>
              <th>explanation</th>
            </tr>
            <tr>
              <td>1</td>
              <td>Louis Armstrong</td>
              <td>trumpet player</td>
              <td>per:occupation</td>
              <td>{}</td>
              <td>The token ',' appears between 'Louis Armstrong' and 'trumpet player':*:*:The token ',' ...</td>
            </tr>
          </table>
          <p>If needed <u>we will use <b>":*:*:"</b> as a separator</u> to split up the string in the explanation
            column. A workaround for the problem of splitting on ","</p>
        </li>
      </ul>
    </div>
    <el-button
        v-if="this.$store.getters.getProjectInfo.task===1 || this.$store.getters.getProjectInfo.task===2 || this.$store.getters.getProjectInfo.task===3"
        @click="downloadAnnotations('json')" type="primary">Download JSON File
    </el-button>
    <el-button v-if="this.$store.getters.getProjectInfo.task===2 || this.$store.getters.getProjectInfo.task===3"
               @click="downloadAnnotations('csv')" type="primary">Download CSV File
    </el-button>
  </div>
</template>

<script>
const fileDownload = require('js-file-download');
//TODO fix style
export default {
  name: "ExportAnnotations",
  methods: {
    downloadAnnotations(format) {
      this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/download_annotations?downloadFormat=${format}`)
          .then((res) => {
            if (format === 'csv') {
              fileDownload(res, `${this.$store.getters.getProjectInfo.name}.${format}`);
            } else if (format === 'json') {
              fileDownload(JSON.stringify(res, null, 1), `${this.$store.getters.getProjectInfo.name}.${format}`);
            }
          })
    }
  }
}
</script>

<style scoped>

</style>
