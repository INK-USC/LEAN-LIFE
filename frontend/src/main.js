import Vue from "vue";
// import Buefy from 'buefy'
// import "buefy/dist/buefy.css";
import ElementUI from 'element-ui';
import "element-ui/lib/theme-chalk/index.css";
// import {BootstrapVue, IconsPlugin} from "bootstrap-vue";
// import 'bootstrap/dist/css/bootstrap.css'
// import 'bootstrap-vue/dist/bootstrap-vue.css'
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
// Vue.use(Buefy);
// Install BootstrapVue
// Vue.use(BootstrapVue);
// Optionally install the BootstrapVue icon components plugin
// Vue.use(IconsPlugin);
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
