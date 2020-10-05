import Vue from "vue";
import ElementUI from 'element-ui';
import "element-ui/lib/theme-chalk/index.css";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import {library} from '@fortawesome/fontawesome-svg-core'
import {faUserSecret} from '@fortawesome/free-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/vue-fontawesome'
import axios from 'axios';

Vue.prototype.$axios = axios;

library.add(faUserSecret);
Vue.component('font-awesome-icon', FontAwesomeIcon)
Vue.config.devtools = true;
Vue.config.productionTip = false;
Vue.use(ElementUI);

Vue.filter('capitalize', (val) => {
	if (!val) return '';
	val = val.toString();
	return val.charAt(0).toUpperCase() + val.slice(1);
})

new Vue({
	router,
	store,
	render: h => h(App)
}).$mount("#app");
