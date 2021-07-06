<template>
  <el-col>
    <!--    <el-autocomplete v-model="documentQuery" placeholder="Search Document" @select="handleDocumentSelected"-->
    <!--                     :fetch-suggestions="searchDocument" style="margin-top: 20px; width: 90%">-->
    <!--      <i slot="prefix" class="el-input__icon el-icon-search"/>-->
    <!--    </el-autocomplete>-->
    <el-input v-model="documentQuery" placeholder="Type to search for documents" style="margin-top: 20px; width: 90%"
              prefix-icon="el-icon-search" clearable/>
    <el-table
        :data="this.$store.getters['document/getDocuments'].documents.filter(row=> !documentQuery || row.text.toLowerCase().includes(documentQuery.toLowerCase().trim()))"
    >
      <el-table-column width="40">
        <template slot-scope="scope">
                  <span v-if="scope.row.annotated">
                    <i class="el-icon-check"/>
                  </span>
          <div v-else>{{ scope.$index + 1 }}</div>
        </template>
      </el-table-column>
      <el-table-column prop="text" :label="tableTitle">
        <template slot-scope="scope">
          <el-link v-line-clamp="3" @click="goToDocument(scope.$index, scope.row)">{{ scope.row.text }}</el-link>
        </template>
      </el-table-column>
    </el-table>

    <el-pagination background layout="prev, pager, next"
                   :total="this.$store.getters['document/getDocuments'].totalDocCount"
                   :page-size="this.$store.getters['document/getDocuments'].pageSize"
                   :current-page="this.$store.getters['document/getDocuments'].curPage"
                   @current-change="pageChanged" style="text-align: center"/>
  </el-col>
</template>

<script>
// the side navigation bar for document
export default {
  name: "AnnotationSideBar",
  data() {
    return {
      documentQuery: "",
    }
  },
  methods: {
    // go to selected document
    goToDocument(index, docInfo) {
      this.$store.dispatch('document/updateCurDocIndex', {curDocIndex: index}, {root: true})
    },

    handleDocumentSelected(item) {
      console.log("document selected", item)
    },
    // searchDocument(_, cb) {
    //   cb([])
    //   if (!this.documentQuery) {
    //     return;
    //   }
    //   this.$http
    //       .get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/?q=${this.documentQuery}`)
    //       .then(res => {
    //         res.results.forEach(row => {
    //           row['value'] = row.text;
    //         })
    //         cb(res.results)
    //
    //         //TODO send res to vuex
    //       })
    // },

    // go to selected page
    pageChanged(newPage) {
      this.$store.dispatch('document/updateCurPage', {newPage: newPage}, {root: true})
    },
  },
  created() {
    this.$store.dispatch('document/fetchDocuments', null, {root: true})
  },
  computed: {
    tableTitle: function () {
      return "Total " + this.$store.getters["document/getDocuments"].totalDocCount + " Documents"
    }
  }

}
</script>

<style scoped>

</style>
