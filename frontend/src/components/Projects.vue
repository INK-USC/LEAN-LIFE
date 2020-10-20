<template>
	<div>
		<h1>
			Hello, {{ this.$store.getters.getUserInfo.username |capitalize }}
		</h1>
		<el-row>
			<el-button type="primary" @click="dialogVisible = true">CREATE PROJECT</el-button>
		</el-row>
		<el-row :gutter="20">
			<el-col :span="12" :offset="6">
				<el-table :data="projects" stripe :default-sort="{prop: 'updated_at', order:'descending'}">
					<el-table-column prop="name" label="Name" sortable/>
					<el-table-column prop="description" label="Description" sortable/>
					<el-table-column prop="task" label="Task" :formatter="convertTaskIDToString" sortable
					                 :filters="this.filters"
					                 :filter-method="filterTask"/>
					<el-table-column prop="updated_at" label="Last Updated" :formatter="dateFormat" sortable/>
					<el-table-column
							label="Operations">
						<template slot-scope="scope">
							<el-button
									size="mini"
									@click="handleEdit(scope.$index, scope.row)">Edit
							</el-button>
							<el-button
									size="mini"
									type="danger"
									@click="handleDelete(scope.$index, scope.row)">Delete
							</el-button>
						</template>
					</el-table-column>
				</el-table>
			</el-col>
		</el-row>
		<CreateProjectModal :dialog-visible.sync="dialogVisible"/>
	</div>

</template>

<script>
import CreateProjectModal from "@/components/CreateProjectModal";
import {PROJECT_TYPE_TO_ID} from "@/utilities/constant";

const {DateTime} = require("luxon");

export default {
	name: "Projects",
	components: {CreateProjectModal},
	data() {
		return {
			projects: [],
			dialogVisible: false,
			filters: []
		}
	},
	created: function () {
		for (let item in PROJECT_TYPE_TO_ID) {
			this.filters = [...this.filters, {text: PROJECT_TYPE_TO_ID[item], value: PROJECT_TYPE_TO_ID[item]}]
		}
		this.$http.get("/projects/").then(
				res => {
					this.projects = res
				}, (err) => {
					console.log(err)
				}
		)
		
	},
	methods: {
		handleEdit(index, row) {
			console.log(index, row);
		},
		handleDelete(index, row) {
			console.log(index, row);
		},
		dateFormat(row, column) {
			let date = row[column.property];
			if (!date) {
				return ""
			}
			return DateTime.fromISO(date).toISODate();
		},
		convertTaskIDToString(row, column) {
			let taskId = row[column.property];
			if (!taskId) {
				return ""
			}
			return PROJECT_TYPE_TO_ID[taskId];
		},
		filterTask(value, row, column) {
			let taskId = row[column.property];
			let taskName = PROJECT_TYPE_TO_ID[taskId];
			return value === taskName
		}
	}
}
</script>

<style scoped>

</style>