<template>
  <el-card>
    <div slot="header" style="display: flex; justify-content: space-between">
      <h3>Historical Annotations</h3>
      <el-link type="primary" @click="cancelUpload">Cancel</el-link>
    </div>
    <div style="text-align: left">
      <div>You can upload a json file of seed annotations to be matched to current unlabeled examples as recommendations
        for the annotator.
      </div>
      <div style="margin-top: 20px">
        You can choose to <b>merge</b> your <b>uploaded</b> file or replace all previously uploaded annotations. Note:
        this will not
        delete previous annotator provided annotations, just the previously uploaded annotations from this process.
      </div>

      <div style="margin-top: 20px">
        <b>It is imperative</b> that the <b>label names match the label space</b> you created. We will lower case and
        match names, so case-sensitivity is not needed.
      </div>

      <div style="margin-top: 20px">
        The format of the file you must upload is as follows:
      </div>
      <pre>
        <code v-if="this.$store.getters.getProjectInfo.task === 2">
{
  "data" : [
    {
      "word" : "Louis Armstrong",
      "label" : "PER"
    },
    {
      "word" : "Corona",
      "label" : "LOC"
    },
    ...
  ]
}
       </code>
        <code v-if="this.$store.getters.getProjectInfo.task === 3">
{
  "data": [
      {
        "word_1" : "Louis Armstrong",
        "word_2" : "trumpet player",
        "label" : "per:occupation"
      },
      ...
  ]
}
      </code>
      </pre>
    </div>
    <div style="text-align: center">
      <el-upload drag accept="text/json" ref="uploadInput" action="" :http-request="uploadFile">
        <i class="el-icon-upload"></i>
        <div class="el-upload__text">Drop file here or <em>click to upload</em></div>
      </el-upload>
      <el-row style="margin-top: 20px">
        <el-radio v-model="action" border label="merge">Merge</el-radio>
        <el-radio v-model="action" border label="replace">Replace</el-radio>
      </el-row>

      <el-table :data="annotations">
        <el-table-column type="index" width="50"/>

        <el-table-column prop="word_1" label="Subject" v-if="this.$store.getters.getProjectInfo.task===3"/>

        <el-table-column prop="word" label="Word" v-if="this.$store.getters.getProjectInfo.task===2"/>
        <el-table-column prop="label" :label="this.$store.getters.getProjectInfo.task===2?'Label': 'Relation'">
          <template slot-scope="scope">
            {{ scope.row.label }}
          </template>
        </el-table-column>

        <el-table-column prop="word_2" label="Object" v-if="this.$store.getters.getProjectInfo.task===3"/>

        <el-table-column label="User Uploaded">
          <template slot-scope="scope">
            {{ scope.row.user_provided }}
          </template>
        </el-table-column>
        <el-table-column>
          <template slot-scope="scope">
            <el-popconfirm title="Are you sure?" @onConfirm="handleDelete(scope.$index, scope.row)"
                           style="margin-left: 10px">
              <el-button size="mini" type="danger" slot="reference"><i class="el-icon-delete"/> Delete</el-button>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination background layout="prev, pager, next" :total="this.pagination.totalDocs"
                     @current-change="pageChanged"/>

      <el-button :disabled="!canGoToAnnotate" :type="canGoToAnnotate?'success':''" style="width: 50%; margin-top: 20px">
        Go to annotate
      </el-button>
    </div>

  </el-card>
</template>

<script>
export default {
  name: "HistoricalAnnotations",
  data() {
    return {
      uploadData: null,
      action: "merge",
      annotations: [],
      labels: [],
      pagination: {
        totalDocs: 0,
        curPage: 1,
      },
      canGoToAnnotate: false,
    }
  },
  methods: {
    uploadFile(param) {
      const formData = new FormData();
      const fileObj = param.file;
      formData.append("history", fileObj);
      formData.append("action", this.action);
      formData.append("task", this.$store.getters.getProjectInfo.task);
      this.$http
          .post(`/projects/${this.$store.getters.getProjectInfo.id}/history/seed/`, formData, {
            headers: {
              ...this.$http.defaults.headers,
              "Content-type": "multipart/form-data"
            }
          })
          .then(res => {
            console.log("upload success", res)
            this.uploadData = null;
            this.action = "merge"
            this.$message({message: "File Uploaded", type: 'success'})
            this.fetchAnnotations()
            this.canGoToAnnotate = true;
          })
          .catch(err => {
            console.log("upload failed", err)
            if (err.response.status === 500) {
              this.$message({
                message: "Sorry, the file you uploaded is not in the right format. Please look above, for the correct format.",
                type: "error"
              })
            }
          })
    },
    fetchAnnotations() {
      this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/history/${this.$store.getters.getProjectInfo.task === 2 ? 'ner' : 're'}/`)
          .then(res => {
            this.pagination.totalDocs = res.count;
            this.annotations = res.results;
            this.annotations.forEach(row => {
              for (let label of this.labels) {
                if (label.id == row.label) {
                  row.label = label.text;
                  break;
                }
              }
            })
          })
    },
    handleDelete(index, row) {
      this.$http.delete(`/projects/${this.$store.getters.getProjectInfo.id}/`)
    },
    cancelUpload() {
      this.$router.back();
    },
    fetchLabels() {
      return this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/labels/`)
          .then(res => {
            this.labels = res;
          })
    },
    pageChanged(pageNum) {
      this.pagination.curPage = pageNum;
      this.fetchAnnotations();
    }
  },
  created() {
    this.fetchLabels()
        .then(() => {
          this.fetchAnnotations()
        })
  },
}
</script>

<style scoped>
pre {
  background-color: rgb(245, 245, 245);
}
</style>
