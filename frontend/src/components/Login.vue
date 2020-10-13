<template>
	<div style="display: flex; justify-content: center">
		<el-col :span="6" v-loading="isLoading">
			<el-form :model="loginForm" :rules="rules" ref="loginForm">
				<el-form-item label="Username" prop="username" class="bold-label">
					<el-input prefix-icon="el-icon-user" v-model="loginForm.username"/>
				</el-form-item>
				<el-form-item label="Password" prop="password" class="bold-label">
					<el-input type="password" prefix-icon="el-icon-lock" v-model="loginForm.password"/>
				</el-form-item>
				<el-form-item>
					<el-checkbox label="Remember me" v-model="loginForm.rememberMe"/>
				</el-form-item>
				<el-form-item>
					<el-button type="primary" @click="login" :disabled=this.isValid style="width: 100%">
						Login
					</el-button>
				</el-form-item>
			</el-form>
		</el-col>
	</div>
</template>

<script>

export default {
	name: "Login",
	data() {
		return {
			loginForm: {
				username: '',
				password: '',
				rememberMe: false,
			},
			rules: {
				username: [{required: true, message: "Please input username", trigger: 'blur'}],
				password: [{required: true, message: "Please input password", trigger: 'blur'}]
			},
			isLoading: false,
			isValid: false,
		}
	},
	methods: {
		login() {
			//TODO connect to backend
			this.$refs['loginForm'].validate(isValid => {
				if (isValid) {
					this.isLoading = true;
					this.$http.post(`/login`, this.loginForm).then(
							res => {
								console.log(res)
								this.$store.commit("login", this.loginForm);
							}, () => {
								this.isLoading = false;
							})

				} else {
					return false;
				}
			});
		}
	},
}
</script>

<style scoped>
.bold-label {
	font-weight: bolder;
}
</style>