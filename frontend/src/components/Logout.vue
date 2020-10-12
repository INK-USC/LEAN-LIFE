<template>
	<div></div>
</template>

<script>
export default {
	name: "Logout.vue",
	created() {
		this.$store.commit("logout", {});
		this.$message({message: "Logout Successfully", type: "warning"});
		this.$router.push("/")
	}
}
</script>

<style scoped>

</style>