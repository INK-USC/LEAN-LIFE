<template>
	<div>
		<el-container>
			<el-aside style="border-right: solid 1px #e6e6e6">
				<SideBar/>
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
import SideBar from "@/components/project/overview/SideBar";

export default {
	name: "Project",
	components: {SideBar, SimplePopup},
}
</script>

<style scoped>

</style>