<template>
  <div>
    <el-container>
      <el-aside style="border-right: solid 1px #e6e6e6">
        <ProjectSideBar v-if="!this.$route.path.endsWith('/annotate')"/>
        <AnnotationSideBar v-if="this.$route.path.endsWith('/annotate')"/>
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
import AnnotationSideBar from "@/components/project/annotation/AnnotationSideBar";
import AnnotationGuidePopup from "@/components/popups/AnnotationGuidePopup";

// placeholder for project page
export default {
  name: "Project",
  components: {AnnotationSideBar, ProjectSideBar, SimplePopup},
}
</script>

<style scoped>

</style>
