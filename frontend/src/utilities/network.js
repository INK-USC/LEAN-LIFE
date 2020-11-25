import axios from 'axios';
import ElementUI from 'element-ui';

axios.defaults.baseURL = process.env.VUE_APP_API_ENDPOINT

const api = axios.create({
	// baseURL: process.env.VUE_APP_API_ENDPOINT,
	headers: {
		'Accept': 'application/json',
		'Content-Type': 'application/json',
		// 'access-control-allow-origin': 'http://0.0.0.0:8000/',
		// "X-CSRFToken": "me9PA22XbQHxfIrCXA8d0f2AIc279g2eiYK1vQdu00sxSVExzXAx2YAFdF4fVbna",
		"X-CSRFToken": "",
		"Authorization": "JWT eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoyLCJ1c2VybmFtZSI6ImppbSIsImV4cCI6MTYwMzIzMjM3OSwiZW1haWwiOiIiLCJvcmlnX2lhdCI6MTYwMzIxNzk3OX0.mxEzxA-qGszb5gsrVOq_PM2XNwBQpPh8etS6eQBcK18"
	}
})


api.interceptors.response.use(
	res => {
		// Any status code that lie within the range of 2xx cause this function to trigger
		// Do something with response data
		return res.data
	},
	err => {
		// Any status codes that falls outside the range of 2xx cause this function to trigger
		// Do something with response error;
		ElementUI.Notification.error(err.toString())
		return Promise.reject(err)
	})

api.interceptors.request.use(
	config => {
		// Do something before request is sent
		if (!config.url.endsWith("/")) {
			ElementUI.Notification.warning("Request not end with slash!")
			// config.url = config.url + "/"
		}
		return config;
	},
	err => {
		// Do something with request error
		ElementUI.Notification.error(err.toString())
		// Message.error("request err: " + err);
		return Promise.reject(err)
	})

export default api;
