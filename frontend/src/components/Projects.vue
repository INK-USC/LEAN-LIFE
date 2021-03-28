<template>
  <div>
    <div style="text-align:center">
      <h1>
        Hello, {{ this.$store.getters.getUserInfo.username | capitalize }}
      </h1>
      <el-row>
        <el-button type="primary" @click="()=> {this.selectedProject=null; this.dialogVisible = true}">CREATE PROJECT
        </el-button>
      </el-row>
    </div>

    <el-row>
      <el-col :span="12" :offset="6">
        <el-table :data="projects" stripe :default-sort="{prop: 'updated_at', order:'descending'}">
          <el-table-column prop="name" label="Name" sortable>
            <template slot-scope="scope">
              <el-link type="primary" @click="handleProjectSelected(scope.$index, scope.row)">
                {{ scope.row.name }}
              </el-link>
            </template>
          </el-table-column>
          <el-table-column prop="description" label="Description" sortable/>
          <el-table-column prop="task" label="Task" :formatter="convertTaskIDToString" sortable
                           :filters="this.filters"
                           :filter-method="filterTask"/>
          <el-table-column prop="updated_at" label="Last Updated" :formatter="dateFormat" sortable/>
          <el-table-column label="Operations">
            <template slot-scope="scope">
              <el-button size="mini" @click="handleEdit(scope.$index, scope.row)"><i class="el-icon-edit"/>
                Edit
              </el-button>
              <el-popconfirm title="Are you sure?" @onConfirm="handleDelete(scope.$index, scope.row)"
                             style="margin-left: 10px">
                <el-button size="mini" type="danger" slot="reference"><i class="el-icon-delete"/>Delete</el-button>
              </el-popconfirm>
            </template>
          </el-table-column>
        </el-table>
      </el-col>
    </el-row>
    <CreateProjectModal :dialog-visible.sync="dialogVisible" :existing-info="this.selectedProject"/>
  </div>

</template>

<script>
import CreateProjectModal from "@/components/CreateProjectModal";
import {ACTION_TYPE, PROJECT_TYPE_TO_ID} from "@/utilities/constant";

const {DateTime} = require("luxon");

export default {
  name: "Projects",
  components: {CreateProjectModal},
  data() {
    return {
      projects: [],
      dialogVisible: false,
      filters: [],
      selectedProject: null,
    }
  },
  created: function () {
    for (let item in PROJECT_TYPE_TO_ID) {
      this.filters = [...this.filters, {text: PROJECT_TYPE_TO_ID[item], value: PROJECT_TYPE_TO_ID[item]}]
    }
    this.fetchProjects();
  },
  methods: {
    handleProjectSelected(index, row) {
      this.$router.push({name: 'DocumentList'})
      this.$store.commit("setProject", {projectInfo: row, actionType: ACTION_TYPE.EDIT});
    },
    handleEdit(index, row) {
      this.selectedProject = row;
      this.dialogVisible = true;
    },
    handleDelete(index, row) {
      this.$http.delete(`/projects/${row.id}/`).then(res => {
        this.fetchProjects();
      })
    },
    fetchProjects() {
      this.$http.get("/projects/").then(
          res => {
            this.projects = res
          }, (err) => {
            console.log(err)
          }
      )
    },
    dateFormat(row, column) {
      let date = row[column.property];
      if (!date) {
        return ""
      }
      return DateTime.fromISO(date).toISODate();
    },
    convertTaskIDToString(row, column) {
      let taskId = row[column.property];
      if (!taskId) {
        return ""
      }
      return PROJECT_TYPE_TO_ID[taskId];
    },
    filterTask(value, row, column) {
      let taskId = row[column.property];
      let taskName = PROJECT_TYPE_TO_ID[taskId];
      return value === taskName
    }
  }
}
</script>

<style>

</style>
