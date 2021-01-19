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
