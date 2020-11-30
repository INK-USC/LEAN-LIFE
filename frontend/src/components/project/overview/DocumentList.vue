<template>
  <div>
    <h1>All Uploaded Documents</h1>
    <el-autocomplete v-model="searchQuery" placeholder="Type to search for documents" style="width: 100%"
                     :fetch-suggestions="searchForDocuments" :debounce="500"/>
    <el-table :data="this.docs" stripe>
      <el-table-column type="index" :index="indexMethod"/>
      <el-table-column prop="text" label="Text"/>
    </el-table>
    <el-pagination background layout="prev, pager, next" :total="this.totalDocs" @current-change="pageChanged"/>
  </div>
</template>

<script>
export default {
  name: "DocumentList",
  data() {
    return {
      docs: [],
      page: 1,
      totalDocs: 0,
      searchQuery: "",
    }
  },
  methods: {
    fetchDocuments() {
      this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/?page=${this.page}`)
          .then(res => {
            this.processResult(res)
          })
    },
    searchForDocuments(_, cb) {
      cb([])
      this.page = 1;
      this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/?page=${this.page}&q=${this.searchQuery}`)
          .then(res => {
            this.processResult(res)
          })
    },
    processResult(res) {
      this.totalDocs = res.count
      this.docs = res.results
    },
    pageChanged(isNext) {
      if (typeof isNext === 'boolean') {
        if (isNext) {
          this.page++;
        } else {
          this.page--;
        }
      } else if (typeof isNext === 'number') {
        this.page = isNext;
      }
      this.fetchDocuments();
    },
    indexMethod(index) {
      return index + 1;
    },
  },
  created() {
    this.fetchDocuments()
  }
}
</script>

<style scoped>

</style>
