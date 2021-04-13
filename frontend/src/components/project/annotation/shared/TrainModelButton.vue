<template>
  <span>
    <el-button @click="trainModelDialogVisible=true">Train Model</el-button>
    <el-dialog title="Train Model" :visible.sync="trainModelDialogVisible">
      <el-form :model="modelForm" label-width="170px">
        <el-form-item required>
          <span slot="label">
            Experiment Name
            <el-popover trigger="hover" :content="paramsExplanations.experiment_name">
              <i class="el-icon-question" slot="reference"/>
            </el-popover>
          </span>
          <el-input v-model="modelForm.experiment_name"/>
        </el-form-item>

        <el-form-item label="Dataset Name" required>
          <span slot="label">
            Dataset Name
            <el-popover trigger="hover" :content="paramsExplanations.dataset_name">
              <i class="el-icon-question" slot="reference"/>
            </el-popover>
          </span>
          <el-input v-model="modelForm.dataset_name"/>
        </el-form-item>

        <el-form-item v-for="(param,index) in Object.keys(modelForm.params)" :key="index" :label="param">
          <span slot="label">
            {{ param }}
            <el-popover trigger="hover" :content="paramsExplanations[param]">
              <i class="el-icon-question" slot="reference"/>
            </el-popover>
          </span>
          <el-select v-if="param==='embeddings'" v-model="modelForm.params[param]" style="width: 100%">
            <el-option v-for="op of defaultParams[param]" :key="op" :label="op" :value="op"/>
          </el-select>

          <el-switch v-else-if="param==='load_model' || param==='soft_match'" v-model="modelForm.params[param]"/>

          <el-input v-else v-model="modelForm.params[param]"/>
        </el-form-item>

        <el-form-item>
          <span slot="label">
            Send Full Text Data?
            <el-popover trigger="hover" :content="paramsExplanations.include_documents">
              <i class="el-icon-question" slot="reference"/>
            </el-popover>
          </span>
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
// train model button to allow user to send parameters and train the model
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
          soft_match: false,
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
        soft_match: false,
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
      },
      paramsExplanations: {
        experiment_name: "unique string to identify the current training experiment being run. If sent with data, data is prepped and saved to match this experiment name, allowing for subsequent trials to use the cached data associated with the experiment name. The experiment name also uniquely identify saved models.",
        dataset_name: "Name of the dataset you are training on, needed for loading of dataset specific variables",
        match_batch_size: "Batch Size for Explanation annotated data",
        unlabeled_batch_size: "Batch Size for Remaining unlabeled data",
        learning_rate: "Downstream classifier learning rat",
        epochs: "Number of epochs downstream classifier will be trained for",
        embeddings: "Embeddings that should be used when building token representations",
        gamma: "Weight between soft and hard classification loss; total_loss = hard + gamma * soft",
        hidden_dim: "Size of hidden vector outputted by downstream BiLSTM classifier in one direction (so full vector size is 2 x hidden_dim)",
        random_state: "Seed to be used for Random functions",
        load_model: "Whether to load a previously saved model, the saved model will be loaded according to the passed in Experiment Name",
        start_epoch: "the epoch to start training at. If you had already trained a model and want the system to load a previously saved model and continue training you can specify the epoch it should start on. ex: you trained model_a for 10 epochs, now you want to train your model_1 for 20, you could tell the system to load model_a and just start training at epoch 11 and you would train for 10 epochs",
        pre_train_hidden_dim: "Size of hidden vector outputted by FIND Module's BiLSTM encoder in one direction (so full vector size is 2 x Pretrain Hidden Dim)",
        pre_train_training_size: "Size of dataset that should be used for pre-training",
        "include_documents": "If data for a particular experiment has already been sent to model training and no new annotations occurred, then there is no need to send data again, as the model training API saves data to be used by subsequent trials tied to the same Experiment Name. If data is sent we clear the pre-saved data associated with an experiment name, re-prepare all data and save the newly sent data for subsequent trials. The first time a trial is run for an experiment data must be sent."
      }

    }
  },
  methods: {
    // submit parameter to backend
    submitTrainingInfo() {
      // console.log("form", this.modelForm)
      const embeddingsArr = this.modelForm.params.embeddings.split(".");
      this.modelForm.params['emb_dim'] = parseInt(embeddingsArr[embeddingsArr.length - 1].replace("d", ""));

      this.$http.post(`/projects/${this.$store.getters.getProjectInfo.id}/train_model/`, this.modelForm).then(res => {
        console.log("train model res", res)
        this.modelNamePopupIsLoading = false
        this.modelNamePopupVisible = false
        this.$notify.success({
          title: `Model ${this.modelForm.experiment_name} training`,
          message: "Your model is being trained now. Click Models in the Navigation Bar to check out its progress"
        })
      }).catch(err => {
        this.$notify.error({
          title: "Model failed to start training",
          message: "Please try again later"
        })
      }).finally(() => {
        this.trainModelDialogVisible = false
      })
    },
    // reset the parameters
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
