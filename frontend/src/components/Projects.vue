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
				<el-table :data="projects" stripe>
					<el-table-column prop="name" label="Name"/>
					<el-table-column prop="name" label="Name"/>
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

export default {
	name: "Projects",
	components: {CreateProjectModal},
	data() {
		return {
			projects: [
				{name: 1, description: "re"},
				{name: 2, description: "re2"}
			],
			dialogVisible: false,
		}
	},
	methods: {
		handleEdit(index, row) {
			console.log(index, row);

		},
		handleDelete(index, row) {
			console.log(index, row);
		},
	}
}
</script>

<style scoped>

</style>