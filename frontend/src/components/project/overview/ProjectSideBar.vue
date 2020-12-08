<template>
  <el-col>
    <el-menu :default-active="$route.path" :router="true">
      <el-submenu index="1">
        <template slot="title">
          <i class="el-icon-receiving"/>
          <span>Corpus</span>
        </template>
        <el-menu-item-group>
          <el-menu-item index="/project/doc/list">
            <i class="el-icon-files"/>
            Overview
          </el-menu-item>
          <el-menu-item index="/project/doc/upload">
            <i class="el-icon-upload"/>
            Upload New Document
          </el-menu-item>
        </el-menu-item-group>
      </el-submenu>

      <el-menu-item index="/project/labels">
        <i class="el-icon-star-on"/>
        <span>Labels</span>
      </el-menu-item>
      <el-menu-item index="/project/settings">
        <i class="el-icon-setting"/>
        <span>Settings</span>
      </el-menu-item>
      <el-menu-item index="/project/annotate">
        <i class="el-icon-edit"/>
        <span>Continue annotating</span>
      </el-menu-item>
    </el-menu>
  </el-col>
</template>

<script>
export default {
  name: "ProjectSideBar"
}
</script>

<style scoped>
.el-menu {
  border-right: none;
}
</style>
