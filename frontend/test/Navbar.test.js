import Vue from "vue";
import NavBar from "../src/components/NavBar";
import { shallowMount, createLocalVue } from '@vue/test-utils'
import VueRouter from 'vue-router'

const localVue = createLocalVue();
localVue.use(VueRouter);
const router= new VueRouter();
shallowMount( NavBar,{
	localVue,
	router
})

describe("Navbar.vue", ()=>{
	it("home page", ()=>{
		const wrapper = mount(NavBar);

		expect(wrapper.vm.$route.path).toBe("/");
		wrapper.find('').element.click();

	})
})