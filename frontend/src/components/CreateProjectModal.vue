<template>
	<el-dialog
			title="Create Project"
			:visible="dialogVisible"
			v-on:update:visible="$emit('update:dialogVisible', $event)"
			width="30%"
	>
		<el-form :model="projectInfo">
			<el-form-item label="Name">
				<el-input v-model="projectInfo.name"/>
			</el-form-item>
			<el-form-item label="Description">
				<el-input v-model="projectInfo.description"/>
			</el-form-item>
			<el-form-item label="Task">
				<el-select remote v-model="projectInfo.task" placeholder="Select">
					<el-option v-for="item in this.taskOptions" :key="item.id" :label="item.name" :value="item.id"/>
				</el-select>
			</el-form-item>
			<el-form-item label="User">
				<el-select remote v-model="projectInfo.users" multiple placeholder="Select">
					<el-option v-for="item in this.users" :key="item.id" :label="item.username" :value="item.id"/>
				</el-select>
			</el-form-item>
		</el-form>
		<span slot="footer" class="dialog-footer">
    <el-button @click="()=>$emit('update:dialogVisible', false)">Cancel</el-button>
    <el-button type="primary"
               @click="createProject">Confirm</el-button>
  </span>
	</el-dialog>
</template>

<script>
export default {
	name: "CreateProjectModal",
	props: {dialogVisible: Boolean},
	data() {
		return {
			taskOptions: [],
			users: [],
			projectInfo: {
				name: "",
				description: "",
				task: "",
				users: [],
			},
		}
	},
	methods: {
		createProject() {
			this.$store.commit("createProject", this.projectInfo);
			this.$emit('update:dialogVisible', false);
			this.$router.push("/create/update")
		}
	},
	created() {
		this.$http.get('/tasks/').then(res => {
			this.taskOptions = res.results
		})
		this.$http.get("/users/").then(res => {
			this.users = res.results
		})
	}
}
</script>

<style scoped>

</style>