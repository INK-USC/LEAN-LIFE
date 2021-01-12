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
            console.log("download for", model, model['file_path'], encodeURIComponent(model['file_path']))

            axios.get("/api/models/download/", {responseType: 'blob', params: {file_path: model['file_path']}})
            .then(res=>{
                console.log("ressss", res)
                    const blob = res.data
                      let url = window.URL.createObjectURL(blob)
                      let a = document.createElement("a");
                      console.log(url);
                      a.href = url;
                      a.download = `${model['model']}.p`;
                      a.click();
            })
        },
        // default is bytes
        toProperSize(size){
            if(!size){
                return "N/A"
            }
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