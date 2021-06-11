<template>
  <el-dialog :visible.sync="dialogVisible">
    <h1 slot="title" style="text-align: center">
      <i class="el-icon-info"/>
      Training Metadata
    </h1>
    <div>
      <pre style="font-size: 20px">{{ this.getMetaData() }}</pre>
    </div>
    <span slot="footer">
      <el-button type="primary" @click="()=>this.$store.commit('hideModelsMetadataPopup')">OK</el-button>
    </span>
  </el-dialog>
</template>

<script>
export default {
  name: "ModelMetaDataPopup",
  data() {
    return {}
  },
  methods: {
    getMetaData() {
      if (this.$store.getters.getModelsMetadataPopupInfo.metadata !== null){
        return JSON.stringify(this.$store.getters.getModelsMetadataPopupInfo.metadata, null, 2);
      }
    }
  },
  computed: {
    dialogVisible: {
      get() {
        return this.$store.getters.getModelsMetadataPopupInfo.dialogVisible
      },
      set() {
        this.$store.commit("hideModelsMetadataPopup")
      }
    },
  }
}
</script>

<style scoped>

</style>
