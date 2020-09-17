class AnnotationDocument {
  constructor(pageNumber, doc, explanationType, projectType) {
    //   currently triggers and nlExplanations are not being used
    this.triggers = {};
    this.nlExplanations = {};
    this.recommendations = null;
    this.pageNumber = pageNumber;
    this.id = doc.id;
    this.text = doc.text;
    this.project = doc.project;
    this.annotated = doc.annotated;
    this.annotations = {};
    this.relationAnnotations = [];
    this.userProvided = doc.user_provided;
    this.hasRecommendations = false;
    this.hasExplanations = false;
    this.projectType = projectType;

    for (let i = 0; i < doc.annotations.length; i++) {
      const ann = doc.annotations[i];
      const extension = ann.extended_annotation;
      const formattedAnnotations = {
        label: ann.label,
        base_ann_id: ann.id,
      };
      if (projectType === 2) {
        formattedAnnotations.start_offset = extension.start_offset;
        formattedAnnotations.end_offset = extension.end_offset;
      } else if (projectType === 3) {
        if (ann.user_provided) {
          formattedAnnotations.type = "ner";
          formattedAnnotations.start_offset = extension.start_offset;
          formattedAnnotations.end_offset = extension.end_offset;
        } else {
          formattedAnnotations.type = "re";
          formattedAnnotations.sbj_start_offset = extension.sbj_start_offset;
          formattedAnnotations.sbj_end_offset = extension.sbj_end_offset;
          formattedAnnotations.obj_start_offset = extension.obj_start_offset;
          formattedAnnotations.obj_end_offset = extension.obj_end_offset;
        }
      }

      this.annotations[ann.id] = formattedAnnotations;

      if (explanationType > 1) {
        this.hasExplanations = true;
        if (explanationType > 2) {
          this.triggers[ann.id] = [[], [], [], []];
          if (ann.explanations) {
            for (let j = 0; j < ann.explanations.length; j++) {
              const subTrigger = ann.explanations[j];
              subTrigger.base_ann_id = ann.id;
              this.triggers[ann.id][subTrigger.trigger_id].push(subTrigger);
            }
          }
        } else {
          this.nlExplanations[ann.id] = [];
          if (ann.explanations) {
            for (let j = 0; j < ann.explanations.length; j++) {
              this.nlExplanations[ann.id].push(ann.explanations[j]);
            }
          }
        }
      }
    }
  }

  addRecommendations(recs) {
    this.recommendations = recs;
    this.hasRecommendations = true;
  }
 
  buildRelation(annotation, id2LabelDict) {
    const docText = this.text;
    let start_offset = annotation.sbj_start_offset;
    let end_offset = annotation.sbj_end_offset;

    const subject = {
      start_offset,
      end_offset,
      text: docText.slice(start_offset, end_offset),
    };

    start_offset = annotation.obj_start_offset;
    end_offset = annotation.obj_end_offset;

    const object = {
      start_offset,
      end_offset,
      text: docText.slice(start_offset, end_offset),
    };

    const relation = id2LabelDict[annotation.label];

    return {
      subject, object, relation, base_ann_id: annotation.base_ann_id,
    };
  }

  extractRelations(id2LabelDict) {
    const annotationList = Object.values(this.annotations);
    for (let i = 0; i < annotationList.length; i++) {
      const annotation = annotationList[i];
      if (annotation.type === 're') {
        this.relationAnnotations.push(this.buildRelation(annotation, id2LabelDict));
      }
    }
  }

  getNerInfoFromRelationAnnotation(key) {
    const annotation = this.annotations[key];
    const nerData = [
      { start_offset: annotation.sbj_start_offset, end_offset: annotation.sbj_end_offset, label: -1 },
      { start_offset: annotation.obj_start_offset, end_offset: annotation.obj_end_offset, label: -1 },
    ];
    return nerData;
  }

  getRERecs() {
    const formattedRecs = {};
    for (let i = 0; i < this.recommendations.length; i++) {
      const rec = this.recommendations[i];
      for (let j = 0; j < rec.length; j++) {
        const r = rec[j]
        if (r["key"] in formattedRecs) {
          formattedRecs[r["key"]].push(r["label"])
        } else {
          formattedRecs[r["key"]] = [r["label"]];
        }
      }
    }
    return formattedRecs;
  }

  addLabelInfoToAnnotation(id2LabelDict) {
    const annIds = Object.keys(this.annotations);
    for (let i = 0; i < annIds.length; i++) {
      this.annotations[annIds[i]].label = id2LabelDict[this.annotations[annIds[i]].label];
    }
  }

  getTriggers(annId) {
    if (annId in this.triggers) {
      return this.triggers[annId];
    }
    return [[], [], [], []];
  }

  getNlExplanations(annId) {
    if (annId in this.nlExplanations) {
      return this.nlExplanations[annId];
    }
    return [];
  }

  getAnnotationList(keys = []) {
    if (keys.length === 0) {
      return Object.values(this.annotations);
    }
    const wantedAnnotations = [];
    for (let i = 0; i < keys.length; i++) {
      wantedAnnotations.push(this.annotations[keys[i]]);
    }
    return wantedAnnotations;
  }

  deleteAnnotation(id) {
    let update = false;
    delete this.annotations[id];
    if (this.triggers.length > 0) {
      delete this.triggers[id];
    } else if (this.nlExplanations.length > 0) {
      delete this.nlExplanations[id];
    }
    if (this.projectType < 3) {
      if (Object.keys(this.annotations).length === 0 && this.annotated) {
        this.annotated = false;
        update = true;
      }
    } else {
      let count = 0;
      const keys = Object.keys(this.annotations);
      for (let i=0; i < keys.length; i++) {
        if (this.annotations[keys[i]].type === "re") {
          count += 1;
        }
      }
      if (count === 0) {
        this.annotated = false;
        update = true;
      }
    }
    return update;
  }
}

export default AnnotationDocument;
