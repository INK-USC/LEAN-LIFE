import axios from 'axios';
import ElementUI from 'element-ui';

axios.defaults.baseURL = process.env.VUE_APP_API_ENDPOINT

const api = axios.create({
	// baseURL: process.env.VUE_APP_API_ENDPOINT,
	headers: {
		'Accept': 'application/json',
		'Content-Type': 'application/json',
		"X-CSRFToken": ""
	}
})

api.interceptors.response.use(
		res => {
			// Any status code that lie within the range of 2xx cause this function to trigger
			// Do something with response data
			return res
		},
		err => {
			// Any status codes that falls outside the range of 2xx cause this function to trigger
			// Do something with response error;
			ElementUI.Notification.error(err.toString())
			return Promise.reject(err)
		})

api.interceptors.request.use(
		res => {
			// Do something before request is sent
			return res;
		},
		err => {
			// Do something with request error
			ElementUI.Notification.error(err.toString())
			// Message.error("request err: " + err);
			return Promise.reject(err)
		})

export default api;