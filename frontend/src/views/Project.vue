<template>
  <div>
    <el-container>
      <el-aside style="border-right: solid 1px #e6e6e6">
        <ProjectSideBar/>
      </el-aside>
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
    <SimplePopup/>
  </div>

</template>

<script>
import SimplePopup from "@/components/popups/SimplePopup";
import ProjectSideBar from "@/components/project/overview/ProjectSideBar";

export default {
  name: "Project",
  components: {ProjectSideBar, SimplePopup},
}
</script>

<style scoped>

</style>
