<template>
	<el-steps :active="$store.getters.getProjectCreatingStep" simple>
		<el-step :title="isCreate? 'Create': 'Edit'" icon="el-icon-plus"></el-step>
		<el-step title="Upload File" icon="el-icon-upload"></el-step>
		<el-step title="Annotate" description="el-icon-pencil"></el-step>
		<el-step title="Step 4" description="Some description"></el-step>
	</el-steps>
</template>

<script>
export default {
	name: "StepBar",
	props: {isCreate: Boolean}
}
</script>

<style scoped>

</style>