export function formatAnnotations(doc, projectType) {
	let res = []
	for (let i = 0; i < doc.annotations.length; i++) {
		const ann = doc.annotations[i];
		const extension = ann.extended_annotation;
		const formattedAnnotation = {
			label: ann.label,
			base_ann_id: ann.id,
		}
		if (projectType === 2) {
			formattedAnnotation.start_offset = extension.start_offset;
			formattedAnnotation.end_offset = extension.end_offset;
		} else if (projectType == 3) {
			if (ann.user_provided) {
				formattedAnnotation.type = "ner";
				formattedAnnotation.start_offset = extension.start_offset;
				formattedAnnotation.end_offset = extension.end_offset;
			} else {
				formattedAnnotation.type = "re";
				formattedAnnotation.sbj_start_offset = extension.sbj_start_offset;
				formattedAnnotation.sbj_end_offset = extension.sbj_end_offset;
				formattedAnnotation.obj_start_offset = extension.obj_start_offset;
				formattedAnnotation.obj_end_offset = extension.obj_end_offset;
			}
		}
		res.push(formattedAnnotation);
	}

	return sortAnnotations(res);
}

function sortAnnotations(annotations) {
	return annotations.sort((a, b) => a.start_offset - b.start_offset);
}

export function formatExplanations(annotation) {
	const explanationsForThisAnnotation = {};

	const explanations = annotation.explanations;
	if (!explanations) {
		return [];
	}
	// console.log("annotation", annotation, "exp", explanations)
	explanations.forEach(exp => {
		const cur = {
			base_ann_id: annotation.id,
			start_offset: exp.start_offset,
			end_offset: exp.end_offset,
			labelId: exp.trigger_id,
			pk_id: exp.id,
		}
		// console.log("exp", cur)
		if (explanationsForThisAnnotation[exp.trigger_id]) {
			explanationsForThisAnnotation[exp.trigger_id].push(cur);
		} else {
			explanationsForThisAnnotation[exp.trigger_id] = [cur];
		}
		// console.log("added", explanationsForThisAnnotation)
	});
	// console.log("all exp", explanationsForThisAnnotation)

	const res = []
	for (let key of Object.keys(explanationsForThisAnnotation)) {
		// console.log("key", key, "value", explanationsForThisAnnotation[key])
		res.push(explanationsForThisAnnotation[key]);
	}
	return res;
}
