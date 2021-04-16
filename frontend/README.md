# How to start frontend

## 1. Install Nodejs

See [Installation Guide](https://nodejs.org/en/download/)

## 2. Install required packages

```
npm install
```

### 3. Compiles and hot-reloads for development

```
npm run serve
```

-----

### Compiles and minifies for production

```
npm run build
```

### Run your unit tests

```
npm run test:unit
```

### Lints and fixes files

```
npm run lint
```

---

# Folder structure

### node_modules

store packages downloaded by [npm](https://www.npmjs.com/).

### public

lean more about `public` folder [here](https://cli.vuejs.org/guide/html-and-static-assets.html#the-public-folder)

### src

actual code for the frontend

### tests

unit testing code

### [.env](https://github.com/motdotla/dotenv#usage)

environmental variable (https://github.com/motdotla/dotenv#usage)

### .eslintrc.js

configuration for [ESLint](https://eslint.org/docs/user-guide/configuring/)

### [.gitignore](https://git-scm.com/docs/gitignore)

files should not be pushed to git listed here

### babel.config.js

configuration for [Babel](https://babeljs.io/docs/en/)

### .jest.config.js

configuration for [Jest](https://jestjs.io/docs/getting-started)

### package.json

required package for the frontend

# Acceptable Behavior

## Tasks

### Sentiment Analysis

- one document can have multiple different label.
- no duplicated label allowed for same document.

### Named Entity Recognition

- one chunk (a word or sentence) can only be annotated by one label.
- one label can be applied to multiple chunks.

### Relation Extraction

- In one relation, subject and object can not be the same chunk.
- Different relation are not related. So one chunk can be the object in relation A, it can also be the object in
  relation B. It can also be subject in relation C.
- no duplication relation (subject, object and label are all the same) is allowed

## Explanations

### Natural Language Explanation

- user should select a template from the dropdown and fill in the blank.
- no duplicate explanation is allowed.

### Trigger Explanation

- one trigger explanation group is one row of explanation.
- one trigger is one selection of chunk.
- one trigger explanation group can have multiple trigger selection.
- no duplicate trigger is allowed within one trigger explanation group.
- Within two different trigger explanation group, the same chunk can be selected more than once.