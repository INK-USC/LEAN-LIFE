import axios from 'axios';
import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
    el: '#models_root',
    delimiters: ["${", "}"],
    data: {
        modelList: []
    },
    methods: {
        fetchModel(){
            axios.get(`/api/models/`).then(res=>{
                this.modelList= res.data;
                this.modelList.forEach((row)=>{
                    row.file_size = this.toProperSize(row.file_size);
                })
            })
        },
        downloadModelClicked(model, index){
            console.log("download for", model, index)
        },
        // default is bytes
        toProperSize(size){
            if(size< 1000){
                return size +" Bytes"
            }else if (size >= 1000  && size< 1000 * 1000){
                return size/1000 + "KB"
            }else if(size>= 1000*1000 && size< 1000 * 1000 * 1000){
                return size/1000000 + "MB"
            }else{
                return size/1000000000+ "GB"
            }
        }
    },
    created(){
        this.fetchModel();
        // refresh every 15 minutes
        setInterval(()=>{
            this.fetchModel();
        }, (15*60)*1000);
    }

})