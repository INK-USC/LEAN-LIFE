<template>
  <div id="app">
    <NavBar/>
    <router-view/>
  </div>
</template>

<style lang="scss">
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 30px;

  a {
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}

div, span, .el-menu-item {
  font-size: 18px;
}

.el-tag {
  font-size: 16px;
}
</style>
<script>
import NavBar from "@/components/NavBar";

export default {
  components: {NavBar},
  created() {
    this.$store.commit("updateAxios")
  }
};
</script>
