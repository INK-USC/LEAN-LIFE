<template>
	<div>
		<el-dialog
				:visible.sync="dialogVisible" width="60%">
			<h1 slot="title">
				<i class="el-icon-info"></i>
				{{ this.dialogContents[this.$store.getters.getSimplePopupInfo.targetDialogType].title }}
			</h1>
			<div v-for="msg in this.dialogContents[this.$store.getters.getSimplePopupInfo.targetDialogType].messages"
			     :key="msg" class="message">
				{{ msg }}
			</div>
			<span slot="footer">
				<el-button type="primary" @click="()=>this.$store.commit('hideSimplePopup')">OK</el-button>
			</span>
		</el-dialog>
	</div>

</template>

<script>
import {DIALOG_TYPE} from "@/utilities/constant";

export default {
	name: "SimplePopup",
	data() {
		return {
			dialogContents: {
				"": {title: "", messages: []}
			}
		}
	},
	created() {
		this.dialogContents[DIALOG_TYPE.UploadDataSet] = {
			title: "Upload Dataset",
			messages: ["Your Project has been created!",
				"You can always go back and edit your project setup on the Project Page.",
				"You must now upload a dataset for your project."]
		}
	},
	computed: {
		dialogVisible: {
			get() {
				return this.$store.getters.getSimplePopupInfo.dialogVisible
			},
			set() {
				this.$store.commit("hideSimplePopup")
			}
		}
	}
	
}
</script>

<style scoped>
.message {
	font-size: 20px;
}
</style>