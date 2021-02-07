<template>
  <div>
    <div style="display: flex; flex-direction: column; align-items: center">
      <div>
        <h1>All Uploaded Documents</h1>
        <el-progress type="circle" :percentage="percentageCompleted" :format="percentageText"/>
      </div>

      <el-input v-model="searchQuery" placeholder="Type to search for documents" style="width: 50%"
                prefix-icon="el-icon-search" clearable/>
    </div>

    <el-table
        :data="this.$store.getters['document/getDocuments'].documents.filter(row=> !searchQuery || row.text.toLowerCase().includes(searchQuery.toLowerCase().trim()))"
        stripe>
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
    pageChanged(pageNum) {
      this.$store.dispatch("document/updateCurPage", {newPage: pageNum}, {root: true})
    },
    indexMethod(index) {
      return index + 1;
    },
    percentageText(percentage) {
      return `${percentage} % annotated`
    }
  },
  created() {
    this.$store.dispatch('document/fetchDocuments', null, {root: true})
  },
  computed: {
    percentageCompleted() {
      const info = this.$store.getters["document/getDocuments"];
      const completed = info.annotatedDocCount;
      const all = info.totalDocCount;
      return (completed / all) * 100;
    }
  }
}
</script>

<style scoped>

</style>
