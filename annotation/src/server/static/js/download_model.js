import axios from 'axios';
import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
    el: '#mail-app',
    delimiters: ["${", "}"],
    data: {
        modelList: []
    },
    methods: {
        fetchModel(){
            axios.get(`/api/projects/104/models`).then(res=>{
                this.modelList= res.data;
            })
        },
        downloadModelClicked(model, index){
           console.log("download for", model, index)
        }
    },
    created(){
        this.fetchModel();
    }
})