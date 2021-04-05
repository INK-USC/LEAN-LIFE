<template>
  <div></div>
</template>

<script>
// logout user
export default {
  name: "Logout.vue",
  created() {
    this.$store.commit("logout", {});
    this.$message({message: "Logout Successfully", type: "warning"});
  }
}
</script>

<style scoped>

</style>
