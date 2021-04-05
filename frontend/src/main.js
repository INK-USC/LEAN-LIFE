import Vue from "vue";
import ElementUI from 'element-ui';
import "element-ui/lib/theme-chalk/index.css";
import App from "./App.vue";
import router from "./router";
import store from "./store";
import {library} from '@fortawesome/fontawesome-svg-core'
import {faUserSecret} from '@fortawesome/free-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/vue-fontawesome'
import Vuex from "vuex";
import api from "@/utilities/network";
import locale from 'element-ui/lib/locale/lang/en'

Vue.prototype.$http = api;

library.add(faUserSecret);
Vue.component('font-awesome-icon', FontAwesomeIcon)
Vue.config.devtools = true;
Vue.config.productionTip = false;
Vue.use(ElementUI, {locale});
Vue.use(Vuex, api);

import lineClamp from 'vue-line-clamp'

// general importing, vue settings
Vue.use(lineClamp, {
	// plugin options
})

Vue.filter('capitalize', (val) => {
	if (!val) return '';
	val = val.toString();
	return val.charAt(0).toUpperCase() + val.slice(1);
})

Vue.filter('displayShortcut', (val) => {
	if (!val) return '';
	val = val.toString().replace('ctrl', 'C').replace('shift', 'S').split(' ').join('-');
	return val;
})

new Vue({
	router,
	store,
	render: h => h(App)
}).$mount("#app");



