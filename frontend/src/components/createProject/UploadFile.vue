<template>
	<el-row>
		<el-col :span="18" :offset="6">
			<el-card>
				<div slot=header style="display: flex">
					<h3>Import your corpus below</h3>
				</div>
				<div style="text-align: left">
					<div>In order to start the annotation process, a corpus <u>must</u> be uploaded</div>
					<div>We accept datasets in the following formats:</div>
					<ul>
						<li>
							<b>JSON (recommended)</b>
							<br/><u>Format:</u>
							<pre>
              <code>
{
  "data" : [
    {
      "text" : "Louis Armstrong the great trumpet player lived in Corona.",
      "foo" : "bar",
      "bar" : "foo"
    },
    {
      "text" : "Spanish Farm Minister Loyola de Palacio had earlier accused Fischler at
                an EU farm ministers' meeting of causing unjustified alarm through
                dangerous generalisation.",
      "foo" : "bar",
      "bar" : "foo"
    },
    ...
  ]
}
              </code>
            </pre>
							Each entry within <i>data</i> must have a key <i><b>text</b></i>. All other keys will be saved in a
							metadata
							dictionary associated
							with the text
						</li>
					</ul>
				</div>
				
				<el-form :model="this.fileForm">
					<el-form-item>
						<el-radio v-model="fileForm.fileType" label="JSON">JSON file</el-radio>
						<el-radio v-model="fileForm.fileType" label="CSV">CSV file</el-radio>
					</el-form-item>
					<el-form-item label="">
						<el-upload :http-request="uploadFile" drag accept="text/json" ref="uploadInput" action="">
							<i class="el-icon-upload"></i>
							<div class="el-upload__text">Drop file here or <em>click to upload</em></div>
						</el-upload>
					</el-form-item>
				</el-form>
			</el-card>
		</el-col>
	</el-row>

</template>

<script>
export default {
	name: "UploadFile",
	data() {
		return {
			fileForm: {
				fileType: "JSON",
				file: new FormData(),
			},
			fileType: ""
		}
	},
	methods: {
		uploadFile(file) {
			console.log("upload file", file.file)
			this.fileForm.file.append("files", file.file);
			//TODO find out how to send file to backend.
		},
	}
}
</script>

<style scoped>

</style>