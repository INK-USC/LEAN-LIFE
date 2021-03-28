<template>
  <span>
    <el-button @click="trainModelDialogVisible=true">Train Model</el-button>
    <el-dialog title="Train Model" :visible.sync="trainModelDialogVisible">
      <el-form :model="modelForm" label-width="160px">
        <el-form-item label="Model Name" required>
          <el-input v-model="modelForm.experiment_name"/>
        </el-form-item>
        <el-form-item label="Dataset Name" required>
          <el-input v-model="modelForm.dataset_name"/>
        </el-form-item>

        <el-form-item v-for="(param,index) in Object.keys(modelForm.params)" :key="index" :label="param">
          <el-select v-if="param==='embeddings'" v-model="modelForm.params[param]" style="width: 100%">
            <el-option v-for="op of defaultParams[param]" :key="op" :label="op" :value="op"/>
          </el-select>

          <el-switch v-else-if="param==='load_model'" v-model="modelForm.params[param]"/>

          <el-input v-else v-model="modelForm.params[param]"/>
        </el-form-item>

        <el-form-item label="Send Full Text Data?">
          <el-switch v-model="modelForm.include_documents">Send full text data?</el-switch>
        </el-form-item>
      </el-form>
      <span slot="footer">
        <el-button @click="trainModelDialogVisible=false">Cancel</el-button>
        <el-button type="primary" @click="submitTrainingInfo">Train Model</el-button>
      </span>
    </el-dialog>
  </span>
</template>


<script>
//TODO need to set condition to disable train model button
export default {
  name: "TrainModelButton",
  data() {
    return {
      trainModelDialogVisible: false,
      modelForm: {
        experiment_name: "",
        dataset_name: "",
        params: {
          match_batch_size: 50,
          unlabeled_batch_size: 100,
          learning_rate: 0.1,
          epochs: 20,
          embeddings: "charngram.100d",
          gamma: 0.5,
          hidden_dim: 100,
          random_state: 42,
          load_model: false,
          start_epoch: 0,
          pre_train_hidden_dim: 300,
          pre_train_training_size: 50000,
        },
        "include_documents": true,
      },
      defaultParams: {
        match_batch_size: 50,
        unlabeled_batch_size: 100,
        learning_rate: 0.1,
        epochs: 20,
        embeddings: ["charngram.100d", "fasttext.en.300d", "fasttext.simple.300d", "glove.42B.300d",
          "glove.840B.300d", "glove.twitter.27B.25d", "glove.twitter.27B.50d", "glove.twitter.27B.100d",
          "glove.twitter.27B.200d", "glove.6B.50d", "glove.6B.100d", "glove.6B.200d", "glove.6B.300d"],
        gamma: 0.5,
        hidden_dim: 100,
        random_state: 42,
        load_model: false,
        start_epoch: 0,
        pre_train_hidden_dim: 300,
        pre_train_training_size: 50000,
      }

    }
  },
  methods: {
    submitTrainingInfo() {
      console.log("form", this.modelForm)
      const embeddingsArr = this.modelForm.params.embeddings.split(".");
      this.modelForm.params['emb_dim'] = parseInt(embeddingsArr[embeddingsArr.length - 1].replace("d", ""));

      this.$http.post('train_model/', this.modelForm).then(res => {
        console.log("train model res", res)
        this.modelNamePopupIsLoading = false
        this.modelNamePopupVisible = false
        this.$notify.success({
          message: `Model ${this.modelForm.experiment_name} training`,
          description: "Your model is being trained now. You can check the status in the models page"
        })
      }).catch(err => {
        this.$notify.error({
          title: "Model failed to start training",
          message: "Please try again later"
        })
      })
    },
    resetToDefault() {
      Object.keys(this.modelForm.params).forEach(param => {
        this.modelForm.params[param] = this.defaultParams[param];
      })
      this.modelForm.include_documents = true;
    }

  },
  created() {
  }
}
</script>

<style scoped>

</style>
