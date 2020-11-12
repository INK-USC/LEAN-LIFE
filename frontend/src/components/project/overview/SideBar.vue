<template>
	<!--	<el-row>-->
	<el-col>
		<el-menu :default-active="$route.path" :router="true">
			<el-submenu index="1">
				<template slot="title">
					<i class="el-icon-receiving"></i>
					<span>Corpus</span>
				</template>
				<el-menu-item-group>
					<el-menu-item index="/project/doc/list">Overview</el-menu-item>
					<el-menu-item index="/project/doc/upload">Upload New Document</el-menu-item>
				</el-menu-item-group>
			</el-submenu>
			
			<el-menu-item index="2">
				<i class="el-icon-star-on"></i>
				<span>Labels</span>
			</el-menu-item>
			<el-menu-item index="3">
				<i class="el-icon-setting"></i>
				<span>Settings</span>
			</el-menu-item>
		</el-menu>
	</el-col>
	
	<!--	</el-row>-->

</template>

<script>
export default {
	name: "SideBar"
}
</script>

<style scoped>
.el-menu {
	border-right: none;
}
</style>