// project type
export const PROJECT_TYPE_TO_ID = {
	1: "Sentiment Analysis",
	2: "Named Entity Recognition",
	3: "Relation Extraction",
};

// template data to show user
export const CSV_TABLE_EXAMPLE_1 = [
	{
		text: "Louis Armstrong the great trumpet player lived in Corona.",
		foo: "bar",
		bar: "foo",
	},
	{
		text: "Spanish Farm Minister Loyola de Palacio had earlier accused Fischler at an EU farm ministers' meeting of causing unjustified alarm through dangerous generalisation.",
		foo: "bar",
		bar: "foo"
	}
]

// steps
export const DIALOG_TYPE = {
	"UploadDataSet": "UploadDataSet",
	"CreatingLabels": "CreatingLabels",
	"ConfiguringOptionalAnnotationSettings": "ConfiguringOptionalAnnotationSettings",
	"Annotation": {
		"SA": "SA",
		"NER": "NER",
		"RE": "RE"
	}
}
// actions
export const ACTION_TYPE = {
	CREATE: "create",
	EDIT: "edit"
}
