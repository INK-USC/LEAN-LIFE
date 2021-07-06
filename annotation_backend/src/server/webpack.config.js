const process = require('process');
const VueLoaderPlugin = require('vue-loader/lib/plugin');
const { ContextReplacementPlugin } = require('webpack');
const BundleTracker = require('webpack-bundle-tracker');
const hljsLanguages = require('./static/js/hljsLanguages');

module.exports = {
    mode: process.env.DEBUG === 'False' ? 'production' : 'development',
    entry: {
        'named_entity_recognition': './static/js/named_entity_recognition.js',
        'relation_extraction': './static/js/relation_extraction.js',
        'sentiment_analysis': './static/js/sentiment_analysis.js',
        'projects': './static/js/projects.js',
        'stats': './static/js/stats.js',
        'label': './static/js/label.js',
        'setting': './static/js/setting.js',
        'upload': './static/js/upload.js',
        'annotationHistory': './static/js/annotationHistory.js',
        'headers': './static/js/headers.js',
        'download': './static/js/download.js',
        'models': './static/js/models.js'
    },
    output: {
        path: __dirname + '/static/bundle',
        filename: '[name].js'
    },
    module: {
        rules: [
            {
                test: /\.vue$/,
                loader: 'vue-loader'
            },
            {
              test: /\.css$/,
              include: /node_modules/,
              loaders: ['style-loader', 'css-loader'],
             }
        ]
    },
    plugins: [
        new ContextReplacementPlugin(
            /highlight\.js\/lib\/languages$/,
            new RegExp(`^./(${hljsLanguages.join('|')})$`)
        ),
        new BundleTracker({ filename: './webpack-stats.json' }),
        new VueLoaderPlugin()
    ],
    resolve: {
        extensions: ['.js', '.vue'],
        alias: {
            vue$: 'vue/dist/vue.esm.js',
        },
    },
}