<template>
	<el-dialog
			:title="this.existingInfo?'Edit Existing Project': 'Create New Project'"
			:visible="dialogVisible"
			v-on:update:visible="$emit('update:dialogVisible', $event)"
			@open="this.dialogIsOpen"
	>
		<create-project-step-bar :isCreate="!this.existingInfo"/>
		<el-form :model="projectInfo">
			<el-form-item label="Name">
				<el-input v-model="projectInfo.name"/>
			</el-form-item>
			<el-form-item label="Description">
				<el-input v-model="projectInfo.description"/>
			</el-form-item>
			<el-form-item label="Task">
				<el-select remote v-model="projectInfo.task" placeholder="Select" style="width: 100%">
					<el-option v-for="item in this.taskOptions" :key="item.id" :label="item.name" :value="item.id"/>
				</el-select>
			</el-form-item>
			<el-form-item label="Explanation Type">
				<el-select remote v-model="projectInfo.explanation_type" placeholder="Select" style="width: 100%">
					<el-option v-for="item in this.explanations" :key="item.id" :label="item.name" :value="item.id"/>
				</el-select>
			</el-form-item>
			<el-form-item label="User">
				<el-select remote v-model="projectInfo.users" multiple placeholder="Select" style="width: 100%">
					<el-option v-for="item in this.users" :key="item.id" :label="item.username" :value="item.id"/>
				</el-select>
			</el-form-item>
		</el-form>
		<span slot="footer" class="dialog-footer">
      <el-button @click="()=>$emit('update:dialogVisible', false)">Cancel</el-button>
      <el-button type="primary" @click="createProject">Confirm</el-button>
    </span>
	</el-dialog>
</template>

<script>

import CreateProjectStepBar from "@/components/project/CreateProjectStepBar";

export default {
	name: "CreateProjectModal",
	components: {CreateProjectStepBar},
	props: {dialogVisible: Boolean, existingInfo: Object},
	data() {
		return {
			taskOptions: [],
			users: [],
			explanations: [],
			projectInfo: {
				name: "",
				description: "",
				guideline: "test",
				task: "",
				explanation_type: "",
				users: [],
			},
		}
	},
	methods: {
		dialogIsOpen() {
			this.$store.commit('updateProjectEditingStep', {step: 0});
			if (this.existingInfo) {
				for (let key in this.existingInfo) {
					this.projectInfo[key] = this.existingInfo[key];
				}
			} else {
				this.projectInfo = this.$store.getters.getEmptyProject
			}
		},
		createProject() {
			let httpRequest;
			if (this.existingInfo) {
				//edit
				httpRequest = this.$http.put(`/projects/${this.projectInfo.id}/`, this.projectInfo)
			} else {
				//create
				httpRequest = this.$http.post("/projects/", this.projectInfo)
			}
			httpRequest.then(res => {
				console.log("create project res", res);
				this.$store.commit("setProject", res);
				this.$emit('update:dialogVisible', false);
				this.$router.push({name: "UploadFile"})
			})
		}
	},
	created() {
		this.$http.get('/tasks/').then(res => {
			this.taskOptions = res.results
		})
		this.$http.get("/users/").then(res => {
			this.users = res.results
		})
		this.$http.get("/explanations/").then(res => {
			res.forEach(row => {
				this.explanations = [...this.explanations, {'id': row[0], 'name': row[1]}]
			})
		})
	},
}
</script>

<style scoped>

</style>