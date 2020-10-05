<template>
	<el-menu :default-active="$route.path" mode="horizontal" @select="handleSelect" router>
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
		<!--		<el-menu-item index="3" disabled>Info</el-menu-item>-->
		<el-menu-item index="/login" v-if="!$store.getters.getUserInfo" class="dock-right">
			Login
		</el-menu-item>
		<el-menu-item index="/logout" v-else class="dock-right" onclick="this.logout">Logout</el-menu-item>
	</el-menu>
</template>

<script>

export default {
	name: "NavBar",
	components: {},
	data() {
		return {
			activeIndex: '1',
			activeIndex2: '1',
			showLogin: false,
			isLoggedIn: false
		};
	},
	methods: {
		handleSelect(key, keyPath) {
			console.log(key, keyPath);
			if (key === "/logout") {
				this.$store.commit("logout");
				this.$message({message: "Logout Successfully", type: "warning"});
			}
		},
	},
}
</script>

<style scoped>
.el-menu > .el-menu-item.dock-right {
	float: right;
}
</style>
