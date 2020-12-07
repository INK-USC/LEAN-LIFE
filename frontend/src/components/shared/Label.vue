<template>
  <div style="display: flex; align-items: center">
    <el-tag class="label-tag" :style="{backgroundColor:labelInfo.background_color, color: labelInfo.text_color}">
      <i class="el-icon-close label-close-icon" @click="removeLabel" v-if="labelInfo.text"/>
      <!--      <el-link>{{ labelInfo.text }}</el-link>-->
      {{ labelInfo.text }}
    </el-tag>
    <span class="el-tag label-keyboard-shortcut"><kbd>{{ labelInfo.shortcut | displayShortcut }}</kbd></span>
  </div>
</template>

<script>
export default {
  name: "Label",
  props: {labelInfo: Object},
  methods: {
    removeLabel() {
      this.$http.delete(`/projects/${this.$store.getters.getProjectInfo.id}/labels/${this.labelInfo.id}`).then(res => {
        this.$store.dispatch('label/fetchLabels', null, {root: true})
      })
    }
  },
  created() {
  }

}
</script>

<style scoped>
.label-close-icon {
  margin-left: -5px;
  margin-right: 7px;
}

.label-tag {
  border-bottom-right-radius: 0px;
  border-top-right-radius: 0px;
}

.label-keyboard-shortcut {
  background: #f5f5f5;
  border-bottom-left-radius: 0px;
  border-top-left-radius: 0px
}

</style>
