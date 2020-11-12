<template>
	<div>
		<h1>
			All Uploaded Documents
		</h1>
		<el-table :data="this.docs" stripe>
			<el-table-column type="index" :index="indexMethod"/>
			<el-table-column prop="text" label="Text"/>
		</el-table>
	</div>
</template>

<script>
export default {
	name: "DocumentList",
	data() {
		return {
			docs: []
		}
	},
	methods: {
		fetchDocuments() {
			this.$http
					.get(`/projects/${this.$store.getters.getProjectInfo.id}/docs/`)
					.then(res => {
						this.docs = res.results
						console.log(res, res.results)
					})
		},
		indexMethod(index) {
			return index + 1;
		}
	},
	created() {
		this.fetchDocuments()
	}
}
</script>

<style scoped>

</style>