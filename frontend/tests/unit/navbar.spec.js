import {createLocalVue, shallowMount} from "@vue/test-utils";
import VueRouter from 'vue-router';
import Vuex from 'vuex';
import ElementUI from 'element-ui';
import NavBar from "@/components/NavBar";

const localVue = createLocalVue()
localVue.use(ElementUI)
localVue.use(VueRouter)
localVue.use(Vuex)

const router = new VueRouter();

describe("Navbar.vue", () => {


	it("renders", () => {
		const getters = {getUserInfo: jest.fn()};
		const store = new Vuex.Store({getters});
		const wrapper = shallowMount(NavBar, {store, localVue, router});
		expect(wrapper.vm.$route.path).toBe("/");
	});
	it("click login", async () => {
		const state = {};
		const mutations = {
			login: jest.fn(),
			logout: jest.fn(),
		};
		const getters = {
			getUserInfo: jest.fn(),
		};
		const store = new Vuex.Store({
			state,
			mutations,
			getters,
		});
		const wrapper = shallowMount(NavBar, {
			store,
			localVue,
			router,
		});
		const loginBtn = wrapper.find('[jest="logBtn"]');
		// console.log(loginBtn.html())
		// loginBtn.trigger('click')
		loginBtn.vm.$emit("click")
		// await wrapper.vm.$nextTick();
		await expect(wrapper.vm.$route.path).toBe('/login')
	});
	it("click logout", async () => {
		const getters = {
			getUserInfo: () => {
				return {username: "jim"}
			},
		};
		const store = new Vuex.Store({getters});
		const wrapper = shallowMount(NavBar, {
			store,
			localVue,
			router,
		});
		const logoutBtn = wrapper.find('[jest="logBtn"]');
		expect(logoutBtn.text()).toBe("Logout")
		logoutBtn.vm.$emit("click")
		await expect(wrapper.vm.$route.path).toBe("/logout")
	})
});
