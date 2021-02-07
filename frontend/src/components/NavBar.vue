<template>
  <el-menu :default-active="$route.path" mode="horizontal" @select="handleSelect" :router="true">
    <el-menu-item index="/">
      <!--			<router-link to="/">-->
      <el-image :src="require('../assets/logo.png')" style="width: 58px; height: 58px"/>
      <!--			</router-link>-->
    </el-menu-item>
    <el-menu-item index="1"><a href="http://inklab.usc.edu/leanlife/" class="el-link" target="_blank">LEAN-LIFE</a>
    </el-menu-item>
    <el-menu-item index="/projects" v-if="$store.getters.getUserInfo">
      Projects
    </el-menu-item>
    <el-menu-item index="/models" v-if="$store.getters.getUserInfo">
      Models
    </el-menu-item>

    <el-menu-item :index="!$store.getters.getUserInfo? '/login': '/logout'" class="dock-right">
      <el-button @click="loginClicked($event)" jest="logBtn">
        {{ !this.$store.getters.getUserInfo ? "Login" : "Logout" }}
      </el-button>
    </el-menu-item>
  </el-menu>
</template>

<script>

export default {
  name: "NavBar",
  components: {},
  data() {
    return {
      showLogin: false,
      isLoggedIn: false
    };
  },
  methods: {
    handleSelect(key, keyPath) {
      console.log(key, keyPath)
    },
    loginClicked() {
      if (!this.$store.getters.getUserInfo) {
        this.$router.push("/login")
      } else {
        this.$router.push("/logout")
      }
    }
  },

}
</script>

<style scoped>
.el-menu > .el-menu-item.dock-right {
  float: right;
}
</style>
