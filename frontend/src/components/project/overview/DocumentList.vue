<template>
  <div>
    <h1>All Uploaded Documents</h1>
    <el-autocomplete v-model="searchQuery" placeholder="Type to search for documents" style="width: 100%"
                     :fetch-suggestions="searchForDocuments" :debounce="500"/>
    <el-table :data="this.$store.getters['document/getDocuments'].documents" stripe>
      <el-table-column type="index" :index="indexMethod"/>
      <el-table-column label="Text">
        <template slot-scope="scope">
          <span v-line-clamp="3">{{ scope.row.text }}</span>
        </template>
      </el-table-column>
    </el-table>
    <el-pagination background layout="prev, pager, next"
                   :total="this.$store.getters['document/getDocuments'].totalDocCount"
                   :page-size="this.$store.getters['document/getDocuments'].pageSize"
                   :current-page="this.$store.getters['document/getDocuments'].curPage"
                   @current-change="pageChanged"/>
  </div>
</template>

<script>
export default {
  name: "DocumentList",
  data() {
    return {
      searchQuery: "",
    }
  },
  methods: {
    searchForDocuments(_, cb) {
      cb([])
      this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/?page=${this.page}&q=${this.searchQuery}`)
          .then(res => {
            //TODO send res to vuex
            // this.processResult(res)
          })
    },
    pageChanged(pageNum) {
      this.$store.dispatch("document/updateCurPage", {newPage: pageNum}, {root: true})
    },
    indexMethod(index) {
      return index + 1;
    },
  },
  created() {
    this.$store.dispatch('document/fetchDocuments', null, {root: true})
  }
}
</script>

<style scoped>

</style>
