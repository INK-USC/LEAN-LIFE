<template>
	<div style="display: flex; justify-content: center">
		<el-form style="width:30%">
			<el-form-item label="Username" class="bold-label">
				<el-input prefix-icon="el-icon-user" v-model="form.name"/>
			</el-form-item>
			<el-form-item label="Password" class="bold-label">
				<el-input prefix-icon="el-icon-lock" v-model="form.password"/>
			</el-form-item>
			<el-form-item>
				<el-checkbox label="Remember me" v-model="form.rememberMe"/>
			</el-form-item>
			<el-form-item>
				<el-button type="primary" @click="login" style="width: 100%">
					Login
				</el-button>
			</el-form-item>

		</el-form>
	</div>
</template>

<script>
export default {
	name: "Login",
	data() {
		return {
			form: {
				name: '',
				password: '',
				rememberMe: false,
			},
		}
	},
	methods: {
		login() {
			//TODO connect to backend
			// this.$axios.get(`${process.env.BASE_URL}/api/login`).then((res) => {
			// 	console.log(res)
			// }).catch(e => {
			// 	// this.error.push(e)
			// 	console.error(e)
			// })
			console.log("my form", this.form)
			this.$store.commit("login", this.form);
			this.$router.push({name: "Projects"})
		}
	}
}
</script>

<style scoped>
.bold-label {
	font-weight: bolder;
}
</style>