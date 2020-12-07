<template>
  <el-col>
    <el-autocomplete v-model="documentQuery" placeholder="Search Document" @select="handleDocumentSelected"
                     :fetch-suggestions="searchDocument" style="margin-top: 20px; width: 90%">
      <i slot="prefix" class="el-input__icon el-icon-search"/>
    </el-autocomplete>

    <el-table :data="this.$store.getters['document/getDocuments'].documents">
      <el-table-column type="index" width="40"/>
      <el-table-column prop="text" :label="tableTitle">
        <template slot-scope="scope">
          <el-link v-line-clamp="2">{{ scope.row.text }}</el-link>
        </template>
      </el-table-column>
    </el-table>
    <el-pagination background layout="prev, pager, next"
                   :total="this.$store.getters['document/getDocuments'].totalDocumentCount"
                   @current-change="pageChanged"/>
  </el-col>
</template>

<script>

export default {
  name: "AnnotationSideBar",
  data() {
    return {
      documentQuery: "",
    }
  },
  methods: {
    handleDocumentSelected(item) {
      console.log("document selected", item)
    },
    searchDocument(_, cb) {
      if (!this.documentQuery) {
        cb([])
        return;
      }
      this.$http
          .get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/?q=${this.documentQuery}`)
          .then(res => {
            res.results.forEach(row => {
              row['value'] = row.text;
            })
            cb(res.results)
          })
    },
    pageChanged(newPage) {
      // console.log("page changed", newPage)
      // this.paginationSetting.curPage = newPage;

      // this.fetchDocuments()
    },
  },
  created() {
    this.$store.dispatch('document/fetchDocuments', null, {root: true})
    // this.fetchDocuments();
  },
  computed: {
    tableTitle: function () {
      return "Total " + this.$store.getters["document/getDocuments"].totalDocumentCount + " Documents"
    }
  }

}
</script>

<style scoped>

</style>
