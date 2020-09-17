import axios from 'axios';

axios.defaults.xsrfCookieName = 'csrftoken';
axios.defaults.xsrfHeaderName = 'X-CSRFToken';
const baseUrl = window.location.href.split('/').slice(3, 5).join('/');
const BASE_HTTP = axios.create({
  baseURL: `/${baseUrl}/`,
});

export default BASE_HTTP;
