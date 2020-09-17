<template>
  <div class="relation-display-wrapper">
    <div class="content">
      <span class="text-sequence tag">{{ subject }}</span>
      <div class="relation-wrapper">
        <div class="relation-text-wrapper">
          <span style="display: inline-block" :style="{ color: relationColor}">{{ relation }}</span>
        </div>
        <div class="arrow-wrapper">
          <div class="arrow-line" :style="{ background: relationColor }"></div>
          <div class="arrow-head" :style="{ borderLeftColor: relationColor }"></div>
        </div>
      </div>
      <span class="text-sequence tag">{{ object }}</span>
    </div>
  </div>
</template>
<script>
module.exports = {
  name: 'relationDisplay',

  props: {
    subject: String,
    object: String,
    relation: String,
    relationColor: String,
  },
};
</script>
