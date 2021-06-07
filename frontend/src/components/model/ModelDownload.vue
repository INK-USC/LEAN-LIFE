<template>
  <div>
    <h1 style="text-align: center">Download Model</h1>

    <el-col :span="20" :offset="2">
      <el-table stripe :data="models">
        <el-table-column type="index"/>
        <el-table-column label="Model Name" prop="model"/>
        <el-table-column label="Project Id" prop="project_id"/>
        <el-table-column label="Project Name" prop="project_name"/>
        <el-table-column label="Project Task" prop="project_task"/>
        <el-table-column label="Standard Trained" prop="is_standard_trained"/>
        <el-table-column label="Failed" prop="is_errored"/>
        <!-- <el-table-column label="Error log" prop="error_message"/> -->
        <el-table-column label="Training Status" prop="training_status"/>
        <el-table-column label="Time Spent" prop="time_spent"/>
        <el-table-column label="Time Left" prop="time_left"/>
        <el-table-column label="Best Training Loss" prop="best_train_loss"/>
        <el-table-column label="File Size" prop="file_size"/>
        <el-table-column label="Actions">
          <template slot-scope="scope">
            <el-button size="mini" @click="downloadModel(scope.row)" :disabled="scope.row.file_size==='N/A'">
              <i class="el-icon-download"/>Download
            </el-button>
            <!--            <el-button size="mini" type="danger" @click="deleteModel(scope.row)">-->
            <!--              <i class="el-icon-delete"/>Delete-->
            <!--            </el-button>-->
          </template>
        </el-table-column>
      </el-table>
    </el-col>
  </div>
</template>

<script>
import fileDownload from 'js-file-download'
// display a list of model user trained
export default {
  name: "ModelDownload",
  data() {
    return {
      models: []
    }
  },
  methods: {
    downloadModel(model) {
      this.$http.get(`/models/download/`, {params: {file_path: model['file_path']}})
          .then(res => {
            fileDownload(res, `${model.model}.p`)
          })
    },
    deleteModel(model) {

    },
    fetchModels() {
      return this.$http.get(`/models/`).then(res => {
        console.log("models", res)
        this.models = res;
      })
    },
  },
  created() {
    this.fetchModels()
    setInterval(() => this.fetchModels().then(() => this.$notify.info({
      title: "Model Training Status have been updated",
      message: ""
    })), 5 * 60 * 1000)

  }
}
</script>

<style scoped>

</style>
