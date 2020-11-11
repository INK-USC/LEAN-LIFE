<template>
	<div>
		<router-view></router-view>
		<SimplePopup/>
	</div>

</template>

<script>
import SimplePopup from "@/components/popups/SimplePopup";

export default {
	name: "Project",
	components: {SimplePopup},
}
</script>

<style scoped>

</style>