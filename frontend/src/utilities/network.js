import axios from 'axios';

axios.interceptors.response.use(
		res => {
			// Any status code that lie within the range of 2xx cause this function to trigger
			// Do something with response data
			return res
		},
		err => {
			// Any status codes that falls outside the range of 2xx cause this function to trigger
			// Do something with response error
			console.error(err);
			this.$message.error("response err: " + err);
			return Promise.reject(err)
		})

axios.interceptors.request.use(
		res => {
			// Do something before request is sent
			return res;
		},
		err => {
			// Do something with request error
			console.error(err);
			this.$message.error("request err: " + err);
			return Promise.reject(err)
		})