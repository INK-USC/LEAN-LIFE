module.exports = {
	preset: "@vue/cli-plugin-unit-jest",
	"transform": {
		"^.+\\.js$": "<rootDir>/node_modules/babel-jest",
		".*\\.(vue)$": "<rootDir>/node_modules/vue-jest"
	},
};
