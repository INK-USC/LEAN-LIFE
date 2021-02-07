<template>
  <div>
    <h1>Download Model</h1>

    <el-table stripe>
      <el-table-column type="index"/>
      <el-table-column label="Model Name"/>
      <el-table-column label="Project Id"/>
      <el-table-column label="Project Name"/>
      <el-table-column label="Project Task"/>
      <el-table-column label="Training Status"/>
      <el-table-column label="Time Spent"/>
      <el-table-column label="Time Left"/>
      <el-table-column label="Best Training Loss"/>
      <el-table-column label="File Size"/>
      <el-table-column label="Actions">
        <template slot-scope="scope">
          <el-button size="mini" @click="downloadModel(scope.row)">
            <i class="el-icon-download"/>
            Download
          </el-button>
          <el-button size="mini" type="danger" @click="deleteModel(scope.row)">
            <i class="el-icon-delete"/>
            Delete
          </el-button>
        </template>
      </el-table-column>

    </el-table>

  </div>
</template>

<script>
export default {
  name: "ModelDownload",
  methods: {
    downloadModel(model) {

    },
    deleteModel(model) {

    },
    fetchModels() {
      this.$http.get(`/models`).then(res => {

      })
    }
  },
  created() {
    this.fetchModels();
  }
}
</script>

<style scoped>

</style>
