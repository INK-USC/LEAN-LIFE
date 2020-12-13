Larger JSON File
{
    "project_type" : 1,
    "label_space" : [{"id": 1, "text" : "label_1_name", "user_provided" : False}, 
                     {"id": 2, "text":"label_2_name", "user_provided" : False}, 
                     {"id": 3, "text":"label_3_name", "user_provided" : True}, 
                     {"id": 4, "text":"label_4_name", "user_provided" : False}],
    "annotated" : [{annotated_doc_1}, {annotated_doc_2}, {annotated_doc_3}]
    "unlabeled" : [{"text" : "abc"}, {"text" : "abc"}, {"text" : "abc"}],
}

Annotated Doc Rep
{
    "text" : "abc",
    "annotations" : [{annotation_1_repr}, {annotation_2_repr}, {annotation_3_repr}, {annotation_4_repr}],
    "explanations" : [{explanation_1_repr}, {explanation_2_repr}]
}

Annotation Reps
SA
{
    "id" : 1,
    "label_text" : "label_1_name",
    "user_provided" : False
}

NER (There will be some of these in RE projects, you can identify by seeing if `user_provided == True`)
{
    "id" : 2,
    "label_text" : "label_2_name",
    "start_offset" : 40, 
    "end_offset" : 80,
    "user_provided" : True
}

RE
{
    "id" : 3,
    "label_text" : "label_3_name",
    "start_offset_1" : 40, 
    "end_offset_1" : 80,
    "start_offset_2" : 10, 
    "end_offset_2" : 20,
    "user_provided" : True
}

Explanation Rep
NL
{
    "annotation_id" : 2
    "text" : "abcd"
}