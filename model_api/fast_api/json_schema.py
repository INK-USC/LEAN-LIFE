"""
    File the contains input and output data schemas for Fast Api Calls

    Allows for data validation to be done by Fast Api, as well as error reporting and interactive documentation
"""
from pydantic import BaseModel
from typing import Any, List, Dict, Tuple, Union, Optional
from typing_extensions import Literal

class SavePathOutput(BaseModel):
    save_path : str

    class Config:
        schema_extra = {
            "examples": {
                "save_path": "/home/ink-lab/LEAN-LIFE/model_api/fast_api/../model_training/internal_api/../next_framework/data/saved_models/Clf_imdb_sa_nle_standard.p"
            }
        }

class LeanLifeParams(BaseModel):
    experiment_name : str
    dataset_name : str
    dataset_size : int
    project_type : str
    project_id: int
    match_batch_size : Optional[str]
    unlabeled_batch_size : Optional[str]
    learning_rate : Optional[float]
    epochs : Optional[int]
    embeddings : Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim : Optional[int]
    gamma : Optional[float]
    hidden_dim : Optional[int]
    random_state : Optional[int]
    load_model : Optional[bool]
    start_epoch : Optional[int]
    pre_train_hidden_dim : Optional[int]
    pre_train_training_size : Optional[int]
    soft_match: Optional[bool]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name" : "test_experiment_1",
                "dataset_name" : "test_dataset",
                "dataset_size" : 3,
                "project_type" : "Relation Extraction"
            }
        }

class TrainingApiParams(BaseModel):
    stage : Literal["both", "clf", "find"]
    experiment_name : str
    dataset_name : str
    dataset_size : int
    task : Literal['sa', 're']
    pre_train_build_data : bool
    build_data : bool
    
    embeddings : Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim : Optional[int]
    custom_vocab_tokens : Optional[List[str]]
    relation_ner_types : Optional[Dict[str, Tuple[str, str]]]

    pre_train_batch_size : Optional[int]
    pre_train_eval_batch_size : Optional[int]
    pre_train_learning_rate : Optional[float]
    pre_train_epochs : Optional[int]
    pre_train_emb_dim : Optional[int]
    pre_train_hidden_dim : Optional[int]
    pre_train_training_size : Optional[int]
    pre_train_random_state : Optional[int]
    pre_train_gamma : Optional[float]
    pre_train_load_model : Optional[bool]
    pre_train_start_epoch : Optional[int]

    match_batch_size : Optional[int]
    unlabeled_batch_size : Optional[int]
    eval_batch_size : Optional[int]
    learning_rate : Optional[float]
    epochs : Optional[int]
    gamma : Optional[float]
    hidden_dim : Optional[int]
    random_state : Optional[int]
    none_label_key : Optional[str]
    load_model : Optional[bool]
    start_epoch : Optional[int]
    eval_data : Optional[List[Tuple[str, str]]]
    soft_match: Optional[bool]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "imdb_sa_nle_soft_match",
                "dataset_name": "imdb_sa_nle_soft_match",
                "dataset_size": 94,
                "task": "sa",
                "match_batch_size": 50,
                "unlabeled_batch_size": 100,
                "learning_rate": 0.1,
                "epochs": 5,
                "embeddings": "charngram.100d",
                "emb_dim": 100,
                "gamma": 0.5,
                "hidden_dim": 100,
                "random_state": 7698,
                "pre_train_hidden_dim": 300,
                "pre_train_training_size": 50000,
                "soft_match": True,
                "stage": "both",
                "pre_train_build_data": True,
                "build_data": True
            }
        }

class EvalApiParams(BaseModel):
    experiment_name : str
    dataset_name : str
    task: str
    train_dataset_size : int
    embeddings : Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim : Optional[int]
    custom_vocab_tokens : Optional[List[str]]
    hidden_dim : Optional[int]
    none_label_key : Optional[str]
    pre_train_training_size : Optional[int]
    eval_batch_size : Optional[int]
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "imdb_sa_nle_soft_match",
                "dataset_name": "imdb_sa_nle_soft_match",
                "train_dataset_size": 94,
                "task": "sa",
                "eval_batch_size": 50,
                "embeddings": "charngram.100d",
                "emb_dim": 100,
                "hidden_dim": 100,
                "pre_train_training_size": 50000
            }
        }

class Label(BaseModel):
    id : int
    text : str
    user_provided : bool

class AnnotatedDoc(BaseModel):
    text : str
    annotations : List[Dict[str, Union[str, bool, int]]]
    explanations: List[Dict[str, Union[str, int]]]

class PlainReDoc(BaseModel):
    text : str
    annotations : List[Dict[str, Union[str, bool, int]]]

class UnlabeledDoc(BaseModel):
    text : str

class LeanLifeData(BaseModel):
    label_space : List[Label]
    annotated : Optional[List[AnnotatedDoc]]
    unlabeled : Optional[List[Union[PlainReDoc, UnlabeledDoc]]]

    class Config:
        schema_extra = {
            "example": {
                "label_space" : [
                    {
                        "id" : 1, "text" : "NER_TYPE_1", "user_provided" : True
                    },
                    {
                        "id" : 2, "text" : "NER_TYPE_2", "user_provided" : True
                    },
                    {
                        "id" : 3, "text" : "NER_TYPE_3", "user_provided" : True
                    },
                    {
                        "id" : 4, "text" : "relation-1", "user_provided" : False
                    },
                    {
                        "id" : 5, "text" : "relation-2", "user_provided" : False
                    }
                ],
                "annotated" : [
                    {
                        "text" : "This is some random text for our relation extraction example, that we're going to insert SUBJ and OBJ into.",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 13, 
                                "end_offset" : 19,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER_TYPE_2",
                                "start_offset" : 53, 
                                "end_offset" : 57,
                                "user_provided" : True
                            },
                            {
                                "id" : 3,
                                "label_text" : "NER_TYPE_3",
                                "start_offset" : 62, 
                                "end_offset" : 69,
                                "user_provided" : True
                            },
                            {
                                "id" : 4,
                                "sbj_start_offset" : 13,
                                "sbj_end_offset" : 19,
                                "obj_start_offset" : 53,
                                "obj_end_offset" : 57,
                                "label_text" : "relation-1",
                                "user_provided" : False
                            },
                            {
                                "id" : 5,
                                "sbj_start_offset" : 62,
                                "sbj_end_offset" : 69,
                                "obj_start_offset" : 53,
                                "obj_end_offset" : 57,
                                "label_text" : "relation-2",
                                "user_provided" : False
                            }
                        ],
                        "explanations" : [
                            {
                                "annotation_id" : 4,
                                "text" : "SUBJ appears to the left of OBJ."
                            },
                            {
                                "annotation_id" : 4,
                                "text" : "The phrase 'random text' appears in the sentence"
                            },
                            {
                                "annotation_id" : 4,
                                "text" : "There is one 1 word between SUBJ and OBJ"
                            },
                            {
                                "annotation_id" : 5,
                                "text" : "The phrase 'example' is to right of SUBJ by at most 7 words."
                            }
                        ]
                    },
                    {
                        "text" : "This document has no relations, but we can still use it for training!",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 5, 
                                "end_offset" : 13,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 21, 
                                "end_offset" : 30,
                                "user_provided" : True
                            },
                            {
                                "id" : 3,
                                "label_text" : "NER_TYPE_3",
                                "start_offset" : 60, 
                                "end_offset" : 68,
                                "user_provided" : True
                            }
                        ],
                        "explanations" : [
                        ]
                    }
                ],
                "unlabeled" : [
                    {
                        "text" : "Some unlabeled text", 
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 0, 
                                "end_offset" : 4,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER_TYPE_2",
                                "start_offset" : 5, 
                                "end_offset" : 14,
                                "user_provided" : True
                            }
                        ]
                    },
                    {
                        "text" : "This won't be considered unlabeled text for relation extraction, too few entities. Though it will be considered in dataset_size.",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 0, 
                                "end_offset" : 4,
                                "user_provided" : True
                            }
                        ]
                    },
                    {"text" : "This also won't be considered unlabeled text for relation extraction. However, for sentiment analysis, all text is considered valid."}
                ]
            }
        }
class LeanLifePayload(BaseModel):
    lean_life_data : LeanLifeData
    params : LeanLifeParams

class ExplanationTriple(BaseModel):
    text : str
    explanation : str
    label : str


class LabeledDoc(BaseModel):
    text: str
    label: str


class ExplanationTrainingPayload(BaseModel):
    params : TrainingApiParams
    label_space : Dict[str, int]
    explanation_triples : Optional[List[ExplanationTriple]]
    unlabeled_text : Optional[List[str]]
    dev_data: Optional[List[LabeledDoc]]
    ner_label_space : Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "imdb_sa_nle_soft_match",
                    "dataset_name": "imdb_sa_nle_soft_match",
                    "dataset_size": 94,
                    "task": "sa",
                    "match_batch_size": 50,
                    "unlabeled_batch_size": 100,
                    "learning_rate": 0.1,
                    "epochs": 5,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "gamma": 0.5,
                    "hidden_dim": 100,
                    "random_state": 7698,
                    "pre_train_hidden_dim": 300,
                    "pre_train_training_size": 50000,
                    "soft_match": True,
                    "stage": "both",
                    "pre_train_build_data": True,
                    "build_data": True
                },
                "label_space": {
                    "Positive": 0,
                    "Negative": 1
                },
                "explanation_triples": [
                    {
                        "text": "Walt Disney's CINDERELLA takes a story everybody's familiar with and embellishes it with humor and suspense, while retaining the tale's essential charm. Disney's artists provide the film with an appealing storybook look that emanates delectable fairy tale atmosphere. It is beautifully, if conventionally, animated; the highlight being the captivating scene where the Fairy Godmother transforms a pumpkin into a majestic coach and Cinderella's rags to a gorgeous gown. Mack David, Al Hoffman, and Jerry Livingston provide lovely songs like A Dream Is a Wish Your Heart Makes\" and \"Bibbidi-Bobbidi-Boo\" that enhance both the scenario and the characters.<br /><br />Even though CINDERELLA's story is predictable",
                        "label": "Positive",
                        "explanation": "The word 'embellishes' appears in the text"
                    },
                    {
                        "text": "I think this is one of the weakest of the Kenneth Branagh Shakespearian works. After such great efforts as Much Ado About Nothing, etc. I thought this was poor. The cast was weaker (Alicia Silverstone, Nivoli, McElhone???) but my biggest gripe was that they messed with the Bard's work and cut out some of the play to put in the musical/dance sequences.<br /><br />You just don't do Shakespeare and then mess with the play. Sorry, but that is just wrong. I love some Cole Porter just like the next person, but jeez, don't mess with the Shakespeare. Skip this and watch Prospero's Books\" if you want to see a brilliant Shakespearean adaptation of the Tempest.\"",
                        "label": "Negative",
                        "explanation": "The word 'weakest' appears in the text and The phrase 'this was poor' appears in the text"
                    },
                    {
                        "text": "You want to know what the writers of this movie consider funny? A robot child sees his robot parents killed (beheaded, as I recall), and then moves between their bodies calling their names. Yeah--what a comic moment. This is the worst movie I ever paid to see.",
                        "label": "Negative",
                        "explanation": "The word 'worst' appears in the text"
                    },
                    {
                        "text": "This was a fantastically written screenplay when it comes to perceiving things from another perspective. The comedy was timely and not overdone, the acting was generally terrific, and the plot line served a greater purpose of generating misconception when we think about people solely based on their external appearance. The plot twists as the brother/sister character of Amanda Bynes tries to play soccer on the boys team finding instead a new love interest along the way. Tatum Channing is where the real misperception lies and he does a fine job of acting disinterested at first, later coming to realize the most important thing in life is friendship, not attitude.",
                        "label": "Positive",
                        "explanation": "The word 'fantastically' appears in the text"
                    },
                    {
                        "text": "Madhur has given us a powerful movie Chandni Bar in the past. His next film Page 3 was one of the worst movies of all time. It apparently tells the story of some high class people in India. After seeing a scene where the man forces another man for sexual reasons to Star in a Movie. I felt like spitting and breaking the DVD. Coincidently i did. The reason why was the movie contains scenes of child pornography and molestation. I literally vomited and was shocked to see a movie showing naked children. Very disturbing stuff, there was no need to show the children fully naked. One of the rich guys likes to kidnap poor children and sell them to foreign people, British men in this movie. I am shocked to know this film was a Hit in parts of India, otherwise Super Flop in UK, USA and Australia. I'm from UK, and this kind of stuff makes me sick, shouldn't of been released in UK.",
                        "label": "Negative",
                        "explanation": "The word 'vomited' appears in the text and The word 'sick' appears in the text and The word 'worst' appears in the text"
                    },
                    {
                        "text": "This remake of the 1962 orginal film'o the book has some very good parts to commend it and some fine performances by some fine actors - however Scorsese opts toward the end for the most formulaic of plot twists and an embarrassingly overacted shakespearean demise that had me looking at my watch.<br /><br />DeNiro is a superb actor, dedicated to giving his all in the work he does, however he needs direction to focus his talent, and this is sorely lacking in the last five minutes of the film.<br /><br />Gregory Peck's cameo is serviceable but nothing more whilst Robert Michum is always fun to watch, even with as few lines as this.<br /><br />Nick Nolte turns in a better performance than Lorenzo's Oil but is not on the same form as Weeds\". Joe Don Baker has some great lines while Juliette Lewis proves yet again that talent sometimes skips a generation.<br /><br />Some good points? The start credits(!)",
                        "label": "Negative",
                        "explanation": "The word 'overacted' appears in the text"
                    }
                ],
                "unlabeled_text": [
                    "The idea of making a film about the Beatles sounds doomed idea, as no production can catch the idea of the actual historic Beatles. Then it is perhaps best not to try to recreate the past, but to produce an illustration that works best with the other available Beatles material. This is exactly what 'Birth of the Beatles' offers to us, the simple story known to us without any extravaganza.<br /><br />*** SPOILERS here on *** <br /><br />Be warned that not everything is that accurate as some Beatles-graduates might expect. The Beatles are seen performing songs that hardly were even composed by that time. The Beatles perform Ask Me Why\"",
                    "There's a certain irony in a parody of the Gothic genre being turned into a mess of clich\u00e9s by filmmakers who either had no idea what the story's purpose was, or just didn't care. All of the hallmarks of your average family film are present- rambunctious younger siblings, a grumpy teenager who doesn't want to move, unsympathetic parents who are unable to see the apparition, and of course a romantic subplot. The movie has very little in common with Wilde's original story, which was largely written to poke fun at the melodramatic Gothic novellas that were all the rage at the time. If Wilde saw this version, he'd probably laugh- and then of course, write a parody. One can only hope that the children who watched this bland, mass-produced pap eventually discovered the wit and sparkle of the original version.",
                    "Lowe returns to the nest after, yet another, failed relationship, to find he's been assigned to jury duty. It's in the plans to, somehow, get out of it, when he realizes the defendant is the girl he's had a serious crush on since the first grade.<br /><br />Through living in the past by telling other people about his feelings towards this girl (played by Camp), Lowe remembers those feelings and does everything in his power to clear Camp of attempted murder, while staying away from the real bad guys at the same time, and succeeding in creating a successful film at the same time.<br /><br />I've heard that St Augustine is the oldest city in the US, and I also know it has some ties to Ponce de Leon, so the backdrop is a good place to start. Unfortunately, it's the only thing good about this movie. The local police are inept, the judge is an idiot, and the defense counsel does everything in her power to make herself look like Joanie Cunningham! I don't know whether to blame the director for poor direction, or for just letting the cast put in such a hapless effort.<br /><br />In short, this movie was so boring, I could not even sleep through it! 1 out of 10 stars!",
                    "The worst film I have seen in the last 12 months. The plot of the story was uninteresting, the movie ended when he became gingesh khan, i always thought there happened something really interesting afterwards. i knew that Mongolia and all the areas where the movie played have beautiful landscapes but the movie didn't profit from that. The jokes where really poor. The narrator, gingesh himself, could have told a bit more about Mongolian history, traditions etc. My co-viewer knew nothing about that at all so he was a bit lost. I was so looking forward to see this film but was really disappointed after all. It was one out of 3 movies I have ever seen in cinema where I considered to leave before the end.",
                    "This movie was highly entertaining. The soundtrack (Bian Adams) is simply beautiful and inspiring. Even more impressive is Brian Adams doing all the songs in French as well. The score is also uplifting and dramatic.<br /><br />The movie is made from a mix of traditional animation, combined with computer generated images. The result is truly stunning. I watch this film at least once a week with my kids and we never tire of it. The story is compelling and well narrated.<br /><br />I don't understand anyone who would rank this movie less than a 7. Definately a keeper in my household.",
                    "Kind of a guilty indulgence nowadays, this used to be required watching when i was in high school. It really is a great illumination of the burgeoning punk scene in LA in 1980. As the bands play, Spheeris prints the lyrics in subtitles, which is of course necessary if one really wants to know what the guy is screaming into the microphone. But also it turns the camera's POV into that of tourist, passing through this alien world. The band interviews reveal an honest approach to the music that really doesn't exist anymore. Then again, it's not as easy to come by $16/month former-church closets like Chavez of Black Flag does. How many unheard of bands do you know that aren't trying like the dickens to get a record deal? These guys just didn't care. And who can't love the commentary of the little French dude who used to be the singer\" for Catholic Discipline (of which Phranc was a member). His gritty voice delivers one of the best soliloquies ever captured on film: \"I have excellent news for the world ... there's no such thing as New Wave.\" Whew! What a relief!\"",
                    "I adored this, but I am an 80's kid. I loved Rainbow Bright my whole childhood. I don't know if little ones these days would be very interested in the show, mine wasn't. (But thats okay, I bought it for me anyway. I just brought the little one so the guy at the checkout stand wouldn't look at me funny.)I love the non violent drama, and the colorful scenery. It just reminds me of a simpler time before cartoons had more violence than our local news can legally show. :) Although I may be just a little biased on the subject. Afterall I was Rainbow Bright 6 years in a row for Halloween........I wonder if they make a Rainbow Bright costume for adults. Lol.",
                    "This is a cute and sad little story of cultural difference. Kyoko is a beautiful Japanese woman who has run to California to escape from a failed relationship in Japan. Ken is a Japanese American manual laborer with aspirations of rock and roll stardom but little concrete to offer a potential partner. Kyoko marries\" Ken in order to be able to stay permanently in the U.S.",
                    "I may have seen worse films than this, but I if I have, I don't remember. Or possibly blocked them out. Who knows,if I was to undergo hypnotherapy, I may remember them, along, maybe, with been abducted by aliens as a child, or other traumas. If so, I would happily exchange those memories for the ones I have of watching this film.<br /><br />I should give the film some credit: It did produce an emotional response. I actually started to become angry at scenes that spoofed other films and TV programs, that this travesty was dirtying them by association. I am terrified that I may be unable to watch films like Dr Strangelove again without this film flitting across my minds eye.",
                    "In Pasadena, Mrs. Davis (Joanna Cassidy) sends her daughter Aubrey Davis (Amber Tamblyn) to Tokyo to bring her sister Karen Davis (Sarah Michelle Gellar), who is interned in a hospital after surviving a fire, back to the USA. After their meeting, Karen dies and Aubrey decides to investigate what happened to her and gets herself cursed in the same situation, being chased by the ghost of the house. Meanwhile in Tokyo, the three high school mates Allison (Arielle Kebbel), Vanessa (Teresa Palmer) and Miyuki (Misako Uno) visit the famous haunted house and are also cursed and chased by the ghost. In Chicago, Trish (Jennifer Beals) moves to the apartment of her boyfriend Bill (Christopher Cousins), who lives with his children, the teenager Lacey (Sarah Roehmer) and boy Jake (Matthew Knight). On the next door, weird things happen with their neighbor.<br /><br />The Grudge 2\" has scary sound and visual effects",
                    "This Norwegian film starts with a man jumping over the subway, apparently committing suicide. But the next scene shows him arriving in a lonely bus into a desert. He meets a man, and is shipped off to a mysterious city, where he starts working in an aseptic modern office as an accountant. The coworkers seem nice, if guarded, he soon meets a girlfriend, yet the city seems utterly strange, as food has no taste, alcohol doesn't make you drunk, and there's nary a children around. Is this a dream, or is he in paradise, or in hell?. While at times, the films looks as extended episode of The Twilight Zone (even at ninety minutes, the movie seems a bit long), it is quite thought provoking. The best scenes are those in which the exaggeration is minimal, as when the people engage in banal conversations about interior decoration, and recoil at discussing deeper issues. I always thought there was something inhuman in advanced capitalist societies, in the way they try to repress the basic urges of human nature. And this movie is best when it devastatingly critiques this life style. Unfortunately, the movie ends up a big long, and the director doesn't seem to know how to end it, but most for of the running time this is very much worth seeing.",
                    "(spoilers?)<br /><br />while the historical accuracy might be questionable... (and with the mass appeal of the inaccurate LOTR.. such things are more easily excused now) I liked the art ness of it. Though not really an art house film. It does provide a little emotionally charged scenes from time to time. <br /><br />I have two complaints. 1. It's too short. and 2. The voice you hear whispering from time to time is not explained.<br /><br />8/10<br /><br />Quality: 10/10 Entertainment: 7/10 Replayable: 5/10",
                    "It's exactly what the title tells you...an island inhabited by fishmen. Shipwrecked doctor Claudio Cassinelli and crew land on the island, they're either picked off by the fishmen or roped into working for treasure hunting lunatic Richard Johnson. Cassinelli discovers that Johnson, who believes he's found the lost city of Atlantis, has been keeping disgraced scientist Joseph Cotten and his daughter Barbara Bach hostage for 15 years so the fishmen can uncover a treasure trove beneath the sea. Cotten, of course, is a complete madman. Bach and Cassinelli have great chemistry. This insanity was directed by Sergio Martino and is not, surprisingly, without merit. It's fast paced, reasonably well acted and the fishmen look pretty convincing (though it's unlikely anyone could prove that these things DON'T look like actual fishmen). There's an excellent music score by Luciano Michelini.",
                ]
            }
        }

class MatchedDataOutput(BaseModel):
    matched_tuples : List[Tuple[str, str]]
    matched_indices : List[Tuple[str, str]]

    class Config:
        schema_extra = {
            "example": {
                "matched_tuples" : [
                    ("At the same time, Chief Financial Officer SUBJ-PERSON will become OBJ-TITLE, succeeding Stephen Green who is leaving to take a government job.", "label-1"),
                    ("Two competing battery makers -- Compact Power Inc. of Troy , Michigan , which is working with parent LG Chem of Korea , and Frankfurt , Germany-based Continental Automotive Systems , which is working with OBJ-ORGANIZATION and SUBJ-ORGANIZATION of Watertown , Massachusetts -- fell 10 weeks behind on delivering the power packs.", "label-2"),
                ],
                "matched_indices" : [
                    (0, 0),
                    (3, 2)
                ]
            }
        }
class StrictMatchPayload(BaseModel):
    explanation_triples : List[ExplanationTriple]
    unlabeled_text : List[str]
    task : Literal["sa", "re"]

    class Config:
        schema_extra = {
            "example": {
                "explanation_triples" : [
                    {
                        "text" : "SUBJ-PERSON 's daughter OBJ-PERSON said Tuesday that her uncle was `` doing very well '' in his lengthy recovery , and was following very closely a gender equality bill under debate.",
                        "explanation" : "The phrase \"'s daughter\" links SUBJ and OBJ and there are no more than three words between SUBJ and OBJ",
                        "label" : "label-1"
                    },
                    {
                        "text" : "SUBJ-PERSON was born OBJ-DATE , in Nashville , Tenn , and graduated with honors from the University of Alabama.",
                        "explanation" : "SUBJ and OBJ sandwich the phrase \"was born\" and there are no more than three words between SUBJ and OBJ",
                        "label" : "label-2"
                    },
                    {
                        "text" : "Under the agreement , AT&T will begin offering SUBJ-ORGANIZATION as part of its OBJ-ORGANIZATION service after Jan. 31 , when AT&T 's current agreement with Dish Network expires.",
                        "explanation" : "There are no more than five words between SUBJ and OBJ and \"as part of its\" appears between SUBJ and OBJ",
                        "label" : "label-3"
                    },
                    {
                        "text" : "SUBJ-PERSON , who died of OBJ-CAUSE_OF_DEATH Monday at the age of 78 , was a complicated person , and any attempt to sum up her life and work will necessarily turn into a string of contradictions.",
                        "explanation" : "Between SUBJ and OBJ the phrase \"who died of\" occurs and there are no more than five words between SUBJ and OBJ",
                        "label" : "label-2"
                    },
                    {
                        "text" : "The style and concept is inspired by three generations of women in their family , with the name `` Der\u00e9on '' paying tribute to SUBJ-PERSON 's grandmother , OBJ-PERSON.",
                        "explanation" : "The phrase \"'s grandmother\" occurs between SUBJ and OBJ and there are no more than four words between SUBJ and OBJ",
                        "label" : "label-4"
                    }
                ],
                "unlabeled_text" : [
                    "At the same time, Chief Financial Officer SUBJ-PERSON will become OBJ-TITLE, succeeding Stephen Green who is leaving to take a government job.",
                    "U.S. District Court Judge OBJ-PERSON in mid-February issued an injunction against Wikileaks after the Zurich-based Bank SUBJ-PERSON accused the site of posting sensitive account information stolen by a disgruntled former employee.",
                    "OBJ-CITY 2009-07-07 11:07:32 UTC French media earlier reported that SUBJ-PERSON , ranked 119 , was found dead by his girlfriend in the stairwell of his Paris apartment.",
                    "Two competing battery makers -- Compact Power Inc. of Troy , Michigan , which is working with parent LG Chem of Korea , and Frankfurt , Germany-based Continental Automotive Systems , which is working with OBJ-ORGANIZATION and SUBJ-ORGANIZATION of Watertown , Massachusetts -- fell 10 weeks behind on delivering the power packs.",
                    "Berkshire shareholders voted Wednesday to split the company 's Class B shares 50-for-1 in a move tied to OBJ-ORGANIZATION 's $ 26.3 billion acquisition of SUBJ-ORGANIZATION.",
                    "The chiefs of more than 60 top companies support the Conservatives ' position and on Thursday the executive chairman of OBJ-NATIONALITY retail giant Marks & Spencer , SUBJ-PERSON , attacked the prime minister for dismissing their concerns."
                ],
                "task" : "re"
            }
        }

class NextEvalDataOutput(BaseModel):
    avg_loss : float
    avg_eval_ent_f1_score : float
    avg_eval_val_f1_score : float
    no_relation_thresholds : Tuple[float, float]

    class Config:
        schema_extra = {
            "example": {
                "avg_loss" : 0.03012,
                "avg_eval_ent_f1_score" : 43.32233,
                "avg_eval_val_f1_score" : 40.5654323,
                "no_relation_thresholds" : (2.912, 0.931)
            }
        }

class EvalNextClfPayload(BaseModel):
    params : EvalApiParams
    label_space : Dict[str, int]
    eval_data : List[Tuple[str, str]]
    ner_label_space: Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "imdb_sa_nle_soft_match",
                    "dataset_name": "imdb_sa_nle_soft_match",
                    "train_dataset_size": 94,
                    "task": "sa",
                    "eval_batch_size": 50,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 100,
                    "pre_train_training_size": 50000
                },
                "label_space": {
                    "Positive": 0,
                    "Negative": 1
                },
                "eval_data": [
                    [
                        "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie",
                        "Positive"
                    ],
                    [
                        "Honestly, this is a very funny movie if you are looking for bad acting (Heather Graham could never live this down... it has three titles for a reason- to protect the guilty!), beautifully bad dialog (\"Do you like... ribs?\"), and a plot only a mother could approve, this is your Friday night entertainment! <br /><br />My roommate rented this under the title \"Terrified\" because he liked Heather Graham, but terrified is what we felt after the final credits. Not because the movie is scary, but because somebody actually paid money to make this turd on a movie reel.<br /><br />Horrible movie. There are a few no-name actors that provide some unintentional comedy, but nothing worth viewing. Heather Graham's dramatic climax also was one of the most pathetic and disturbing things I have ever witnessed. I award this movie no point, and may God have mercy on its soul.",
                        "Negative"
                    ],
                    [
                        "This third Darkman was definitely better than the second one, but still far worse than the original movie. What made this one better than D2 was the fact that The Bad Guy had been changed and Durant was not brought back again. Furthermore there was actually some hint of character development when it came to the bad guy's family and Darkman himself. This made my heart soften and I gave this flick as much as 4/10, i.e. **/*****.",
                        "Negative"
                    ],
                    [
                        "surely this film was hacked up by the studio? perhaps not but i feel there were serious flaws in the storytelling that if not attributed to the editing process could only be caused by grievously bad, criminal indeed, writing and directing.<br /><br />i understand the effect burton wished to achieve with the stylised acting similar to the gothic fairytale atmosphere of edward scissorhands, but here unfortunately it falls flat and achieves no mythical depth of tropes but only the offensive tripe of affectation. ie bad acting and shallow characterisation even for a fairytale.<br /><br />finally not that scary, indeed only mildly amusing in its attempts. the use of dialogue as a vehicle for plot background was clumsy and unnecessary. the mystery of who is the headless horseman would suffice, no need for the myth about a german mercenary, although christopher walken did cut a dashing figure but not that menacing - seeing the horsemans head makes him seem far friendlier that a decapitated inhuman nine foot tall spirit as in the original legend.<br /><br />no real rhythm or universal tone was ever established and not a classic in burtons oevure. stilted and clipped as my parting shot...",
                        "Negative"
                    ],
                    [
                        "The \"good news\" is that the circus is in town. The \"bad news\" is that's right over Bugs Bunny's underground home. He wakes up as his place shakes like an earthquake hit it, when workers pound stakes into the ground and elephants stomp by, etc.<br /><br />To be more specific, the lions' cage is place exactly over Bugs' hole. The lion sniffs food, and by process of elimination, figures out it's a rabbit. Bugs, curious what all the racket is about, winds his way through the tunnel and winds up in the lion's mouth.<br /><br />I'll say for thing for BB: he is totally fearless, at least in this cartoon, and at least for 30 seconds. When he comes to his senses, he runs like crazy and we get a lion-versus-a rabbit battle the rest of the way. Once again, Bugs faces dumb opponent, one he calls \"Nero,\" but lion is fierce and Bugs will need all his wits and somewhat-fake bravado to fend off this beast.<br /><br />About half the gags are stupid and the other half funny, but always fast-moving, colorful and good enough to recommend. I mean, it's not everyday you can see a lion on a trapeze, or doing a hula dance!",
                        "Positive"
                    ],
                    [
                        "I never thought I would absolutly hate an Arnold Schwartzeneggar film, BUT this is is dreadful from the get go. there isnt one redeemable scene in the entire 123 long minutes. an absolute waste of time<br /><br /> thank yu<br /><br /> Jay harris",
                        "Negative"
                    ],
                    [
                        "Before I explain the \"Alias\" comment let me say that \"The Desert Trail\" is bad even by the standards of westerns staring The Three Stooges. In fact it features Carmen Laroux as semi- bad girl Juanita, when you hear her Mexican accent you will immediately recognize her as Senorita Rita from the classic Stooge short \"Saved by the Belle\". <br /><br />In \"The Desert Trail\" John Wayne gets to play the Moe Howard character and Eddy Chandler gets to play Curly Howard. Like their Stooge counterparts a running gag throughout the 53- minute movie is Moe hitting Curly. Wayne's character, a skirt chasing bully, is not very endearing, but is supposed to be the good guy. <br /><br />Playing a traveling rodeo cowboy Wayne holds up the rodeo box office at gunpoint and takes the prize money he would have won if the attendance proceeds had been good-the other riders have to settle for 25 cents on the dollar (actually even less after Wayne robs the box office). No explanation is given for Wayne's ripping off the riders and still being considered the hero who gets the girl. <br /><br />Things get complicated at this point because the villain (Al Ferguson) and his sidekick Larry Fine (played by Paul Fix-who would go on to play Sheriff Micah on television's \"The Rifleman\") see Wayne rob the box office and then steal the remainder of the money and kill the rodeo manager. Moe and Curly get blamed. <br /><br />So Moe and Curly move to another town to get away from the law and they change their names to Smith and Jones. Who do they meet first but their old friend Larry, whose sister becomes the 2nd half love interest (Senorita Rita is left behind it the old town and makes no further appearances in the movie). <br /><br />Larry's sister is nicely played by a radiantly beautiful Mary Kornman (now grown up but in her younger days she was one of the original cast members of Hal Roach's \"Our Gang\" shorts). Kornman is the main reason to watch the mega-lame western and her scenes with Moe and Curly are much better than any others in the production, as if they used an entirely different crew to film them. <br /><br />Even for 1935 the action sequences in this thing are extremely weak and the technical film- making is staggeringly bad. The two main chase scenes end with stock footage wide shots of a rider falling from a horse. Both times the editor cuts to a shot of one of the characters rolling on the ground, but there is no horse in the frame, the film stock is completely different, and the character has on different clothes than the stunt rider. There is liberal use of stock footage in other places, none of it even remotely convincing. <br /><br />One thing to watch for is a scene midway into the movie where Moe and Curly get on their horses and ride away (to screen right) from a cabin as the posse is galloping toward the cabin from the left. The cameraman follows the two stooges with a slow pan right and then does a whip pan to the left to reveal the approaching posse. Outside of home movies I have never seen anything like this, not because it is looks stupid (which it does) but because a competent director would never stage a scene in this manner. They would film the two riders leaving and then reposition the camera and film the posse approaching as a separate action. Or if they were feeling creative they would stage the sequence so the camera shows the riders in the foreground and the posse approaching in the background. <br /><br />Then again, what do I know? I'm only a child.",
                        "Negative"
                    ],
                    [
                        "i am rarely moved to make these kind of comments BUT after sitting through most of rankin's dreadful movie i feel like i have really earned the right to say what i feel about it! i couldn't actually make it right to the end, and became one of the half dozen or more walk outs (about 1/3rd of the audience) after the ragged plot, woeful dialogue and insulting characterisation became just too much to bear. this film is all pose and no art. all style and no substance. it is weighed down by dreadful acting, a genuinely dire script, indifferent cinematography and student-level production values. how it got funded, started, and finished is a mystery to me. i bet you a million quid it never goes on general release. the proper critics would tear it apart. a really bad film. shockingly bad. a really really really poor effort AND that is without even mentioning the gratuitous new-born-kitten-gets-dropped-into-a-deep-fat-fryer moment. totally meaningless, utterly lightweight, poorly put together; this movie is a dreadful embarrassment for uk cinema.",
                        "Negative"
                    ],
                    [
                        "I watched this movie for the hot guy--and even he sucked! He was the worst one--well, okay, I have to give props to that freaky police officer rapist guy too, he was even worse. The guy wasn't that cute in the end, he had the most terrible accent, and he was the most definite definition of hicksville idiot that can't stand up to his mom for the one he \"loves\" there's ever been. Overall, and if this makes any sense to you, when I go to pick up movies at the video store, I think to myself as I read the back of a movie that looks so/so, \"Well, at least it can't be worse than Carolina Moon.\" The most terrible movie, and the most terrible writing, acting, plot--everything in it made my gag reflexes want to do back flips. It was THE most horrid movie I will ever see, with Gabriela way up there too. I hated it, and trust me, if there was any number under 1 IMDb had for rating, I'd choose that in a heartbeat.",
                        "Negative"
                    ],
                    [
                        "Stargate SG-1 is a spin off of sorts from the 1994 movie \"Stargate.\" I am so glad that they decided to expand on the subject. The show gets it rolling from the very first episode, a retired Jack O'Neill has to go through the gate once more to meet with his old companion, Dr. Daniel Jackson. Through the first two episodes, we meet Samantha Carter, a very intelligent individual who lets no one walk over her, and there is Teal'c, a quiet, compassionate warrior who defies his false god and joins the team. <br /><br />The main bad guys are called the Gouald, they are parasites who can get inserted into one's brain, thus controlling them and doing evil deeds. Any Gouald who has a massive amount of power is often deemed as a \"System Lord.\" The warriors behind the Gouald are called Jaffa, who house the parasitic Gouald in their bodies until the Gouald can get inserted in a person's brain.<br /><br />Through the episodes, we mostly get to see SG-1, the exploratory team comprised of Jack/Daniel/Teal'c/and Sam, go through the wormhole that instantly transports them to other planets (this device is called the Stargate) and they encounter new cultures or bad guys. Some episodes are on-world, meaning that they do not go through the Stargate once in the episode and rather deal with pressing issues on Earth.<br /><br />Through the years, you start to see a decline in the SG-1 team as close knit, and more character-building story lines. This, in turn means even more on-world episodes, which is perfectly understandable.<br /><br />My rating: 8.75/10----While most of this show is good, there are some instances of story lines not always getting wrapped up and less of an emphasis on gate travel these last few years. But still, top notch science fiction!",
                        "Positive"
                    ]
                ]
            }
        }

class SoftMatchData(BaseModel):
    scores: List[List[str]]


class StandardTrainingApiParams(BaseModel):
    experiment_name: str
    dataset_name: str
    task: Literal['sa', 're', 'ner']
    build_data: bool

    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    custom_vocab_tokens: Optional[List[str]]
    relation_ner_types: Optional[Dict[str, Tuple[str, str]]]

    match_batch_size: Optional[int]
    eval_batch_size: Optional[int]
    learning_rate: Optional[float]
    epochs: Optional[int]
    random_state: Optional[int]
    emb_dim: Optional[int]
    hidden_dim: Optional[int]
    none_label_key: Optional[str]
    load_model: Optional[bool]
    start_epoch: Optional[int]
    eval_data: Optional[List[Tuple[str, str]]]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "imdb_sa_nle_standard",
                "dataset_name": "imdb_sa_nle_standard",
                "task": "sa",
                "match_batch_size": 50,
                "learning_rate": 0.1,
                "epochs": 5,
                "embeddings": "charngram.100d",
                "emb_dim": 100,
                "hidden_dim": 100,
                "random_state": 7698,
                "build_data": True
            }
        }


class StandardPipelinePayload(BaseModel):
    params: StandardTrainingApiParams
    label_space: Dict[str, int]
    labeled_data: List[LabeledDoc]
    dev_data: Optional[List[LabeledDoc]]
    ner_label_space: Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "imdb_sa_nle_standard",
                    "dataset_name": "imdb_sa_nle_standard",
                    "task": "sa",
                    "match_batch_size": 50,
                    "learning_rate": 0.1,
                    "epochs": 5,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 100,
                    "random_state": 7698,
                    "build_data": True
                },
                "label_space": {
                    "Positive": 0,
                    "Negative": 1
                },
                "labeled_data": [
                    {
                        "text": "Walt Disney's CINDERELLA takes a story everybody's familiar with and embellishes it with humor and suspense, while retaining the tale's essential charm. Disney's artists provide the film with an appealing storybook look that emanates delectable fairy tale atmosphere. It is beautifully, if conventionally, animated; the highlight being the captivating scene where the Fairy Godmother transforms a pumpkin into a majestic coach and Cinderella's rags to a gorgeous gown. Mack David, Al Hoffman, and Jerry Livingston provide lovely songs like A Dream Is a Wish Your Heart Makes\" and \"Bibbidi-Bobbidi-Boo\" that enhance both the scenario and the characters.<br /><br />Even though CINDERELLA's story is predictable",
                        "label": "Positive"
                    },
                    {
                        "text": "I think this is one of the weakest of the Kenneth Branagh Shakespearian works. After such great efforts as Much Ado About Nothing, etc. I thought this was poor. The cast was weaker (Alicia Silverstone, Nivoli, McElhone???) but my biggest gripe was that they messed with the Bard's work and cut out some of the play to put in the musical/dance sequences.<br /><br />You just don't do Shakespeare and then mess with the play. Sorry, but that is just wrong. I love some Cole Porter just like the next person, but jeez, don't mess with the Shakespeare. Skip this and watch Prospero's Books\" if you want to see a brilliant Shakespearean adaptation of the Tempest.\"",
                        "label": "Negative"
                    },
                    {
                        "text": "You want to know what the writers of this movie consider funny? A robot child sees his robot parents killed (beheaded, as I recall), and then moves between their bodies calling their names. Yeah--what a comic moment. This is the worst movie I ever paid to see.",
                        "label": "Negative"
                    },
                    {
                        "text": "This was a fantastically written screenplay when it comes to perceiving things from another perspective. The comedy was timely and not overdone, the acting was generally terrific, and the plot line served a greater purpose of generating misconception when we think about people solely based on their external appearance. The plot twists as the brother/sister character of Amanda Bynes tries to play soccer on the boys team finding instead a new love interest along the way. Tatum Channing is where the real misperception lies and he does a fine job of acting disinterested at first, later coming to realize the most important thing in life is friendship, not attitude.",
                        "label": "Positive"
                    },
                    {
                        "text": "Madhur has given us a powerful movie Chandni Bar in the past. His next film Page 3 was one of the worst movies of all time. It apparently tells the story of some high class people in India. After seeing a scene where the man forces another man for sexual reasons to Star in a Movie. I felt like spitting and breaking the DVD. Coincidently i did. The reason why was the movie contains scenes of child pornography and molestation. I literally vomited and was shocked to see a movie showing naked children. Very disturbing stuff, there was no need to show the children fully naked. One of the rich guys likes to kidnap poor children and sell them to foreign people, British men in this movie. I am shocked to know this film was a Hit in parts of India, otherwise Super Flop in UK, USA and Australia. I'm from UK, and this kind of stuff makes me sick, shouldn't of been released in UK.",
                        "label": "Negative"
                    },
                    {
                        "text": "This remake of the 1962 orginal film'o the book has some very good parts to commend it and some fine performances by some fine actors - however Scorsese opts toward the end for the most formulaic of plot twists and an embarrassingly overacted shakespearean demise that had me looking at my watch.<br /><br />DeNiro is a superb actor, dedicated to giving his all in the work he does, however he needs direction to focus his talent, and this is sorely lacking in the last five minutes of the film.<br /><br />Gregory Peck's cameo is serviceable but nothing more whilst Robert Michum is always fun to watch, even with as few lines as this.<br /><br />Nick Nolte turns in a better performance than Lorenzo's Oil but is not on the same form as Weeds\". Joe Don Baker has some great lines while Juliette Lewis proves yet again that talent sometimes skips a generation.<br /><br />Some good points? The start credits(!)",
                        "label": "Negative"
                    }
                ]
            }
        }


class LeanLifeStandardParams(BaseModel):
    experiment_name: str
    dataset_name: str
    project_type: str
    project_id: int

    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]

    match_batch_size: Optional[int]
    learning_rate: Optional[float]
    epochs: Optional[int]
    random_state: Optional[int]
    emb_dim: Optional[int]
    hidden_dim: Optional[int]
    load_model: Optional[bool]
    start_epoch: Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "test_experiment_1",
                "dataset_name": "test_dataset",
                "project_type": "Relation Extraction"
            }
        }


class StandardAnnotatedDoc(BaseModel):
    text: str
    annotations: List[Dict[str, Union[str, bool, int]]]


class LeanLifeStandardData(BaseModel):
    label_space: List[Label]
    labeled: Optional[List[StandardAnnotatedDoc]]

    class Config:
        schema_extra = {
            "example": {
                "label_space" : [
                    {
                        "id" : 1, "text" : "NER_TYPE_1", "user_provided" : True
                    },
                    {
                        "id" : 2, "text" : "NER_TYPE_2", "user_provided" : True
                    },
                    {
                        "id" : 3, "text" : "NER_TYPE_3", "user_provided" : True
                    },
                    {
                        "id" : 4, "text" : "relation-1", "user_provided" : False
                    },
                    {
                        "id" : 5, "text" : "relation-2", "user_provided" : False
                    }
                ],
                "annotated" : [
                    {
                        "text" : "This is some random text for our relation extraction example, that we're going to insert SUBJ and OBJ into.",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 13, 
                                "end_offset" : 19,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER_TYPE_2",
                                "start_offset" : 53, 
                                "end_offset" : 57,
                                "user_provided" : True
                            },
                            {
                                "id" : 3,
                                "label_text" : "NER_TYPE_3",
                                "start_offset" : 62, 
                                "end_offset" : 69,
                                "user_provided" : True
                            },
                            {
                                "id" : 4,
                                "sbj_start_offset" : 13,
                                "sbj_end_offset" : 19,
                                "obj_start_offset" : 53,
                                "obj_end_offset" : 57,
                                "label_text" : "relation-1",
                                "user_provided" : False
                            },
                            {
                                "id" : 5,
                                "sbj_start_offset" : 62,
                                "sbj_end_offset" : 69,
                                "obj_start_offset" : 53,
                                "obj_end_offset" : 57,
                                "label_text" : "relation-2",
                                "user_provided" : False
                            }
                        ]
                    },
                    {
                        "text" : "This document has no relations, but we can still use it for training!",
                        "annotations" : [
                            {
                                "id" : 1,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 5, 
                                "end_offset" : 13,
                                "user_provided" : True
                            },
                            {
                                "id" : 2,
                                "label_text" : "NER_TYPE_1",
                                "start_offset" : 21, 
                                "end_offset" : 30,
                                "user_provided" : True
                            },
                            {
                                "id" : 3,
                                "label_text" : "NER_TYPE_3",
                                "start_offset" : 60, 
                                "end_offset" : 68,
                                "user_provided" : True
                            }
                        ]
                    }
                ]
            }
        }


class LeanLifeStandardPayload(BaseModel):
    lean_life_data: LeanLifeStandardData
    params: LeanLifeStandardParams


class EvalStandardApiParams(BaseModel):
    experiment_name: str
    dataset_name: str
    task: Literal['sa', 're', 'ner']

    eval_batch_size: Optional[int]
    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim: Optional[int]
    hidden_dim: Optional[int]
    none_label_key: Optional[str]

    custom_vocab_tokens: Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "imdb_sa_nle_standard",
                "dataset_name": "imdb_sa_nle_standard",
                "task": "sa",
                "eval_batch_size": 50,
                "embeddings": "charngram.100d",
                "emb_dim": 100,
                "hidden_dim": 100
            }
        }


class EvalStandardClfPayload(BaseModel):
    params: EvalStandardApiParams
    ner_label_space: Optional[List[str]]
    label_space: Dict[str, int]
    eval_data: List[Tuple[str, str]]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "imdb_sa_nle_standard",
                    "dataset_name": "imdb_sa_nle_standard",
                    "task": "sa",
                    "eval_batch_size": 50,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 100
                },
                "label_space": {
                    "Positive": 0,
                    "Negative": 1
                },
                "eval_data": [
                    [
                        "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie",
                        "Positive"
                    ],
                    [
                        "Honestly, this is a very funny movie if you are looking for bad acting (Heather Graham could never live this down... it has three titles for a reason- to protect the guilty!), beautifully bad dialog (\"Do you like... ribs?\"), and a plot only a mother could approve, this is your Friday night entertainment! <br /><br />My roommate rented this under the title \"Terrified\" because he liked Heather Graham, but terrified is what we felt after the final credits. Not because the movie is scary, but because somebody actually paid money to make this turd on a movie reel.<br /><br />Horrible movie. There are a few no-name actors that provide some unintentional comedy, but nothing worth viewing. Heather Graham's dramatic climax also was one of the most pathetic and disturbing things I have ever witnessed. I award this movie no point, and may God have mercy on its soul.",
                        "Negative"
                    ],
                    [
                        "This third Darkman was definitely better than the second one, but still far worse than the original movie. What made this one better than D2 was the fact that The Bad Guy had been changed and Durant was not brought back again. Furthermore there was actually some hint of character development when it came to the bad guy's family and Darkman himself. This made my heart soften and I gave this flick as much as 4/10, i.e. **/*****.",
                        "Negative"
                    ],
                    [
                        "surely this film was hacked up by the studio? perhaps not but i feel there were serious flaws in the storytelling that if not attributed to the editing process could only be caused by grievously bad, criminal indeed, writing and directing.<br /><br />i understand the effect burton wished to achieve with the stylised acting similar to the gothic fairytale atmosphere of edward scissorhands, but here unfortunately it falls flat and achieves no mythical depth of tropes but only the offensive tripe of affectation. ie bad acting and shallow characterisation even for a fairytale.<br /><br />finally not that scary, indeed only mildly amusing in its attempts. the use of dialogue as a vehicle for plot background was clumsy and unnecessary. the mystery of who is the headless horseman would suffice, no need for the myth about a german mercenary, although christopher walken did cut a dashing figure but not that menacing - seeing the horsemans head makes him seem far friendlier that a decapitated inhuman nine foot tall spirit as in the original legend.<br /><br />no real rhythm or universal tone was ever established and not a classic in burtons oevure. stilted and clipped as my parting shot...",
                        "Negative"
                    ],
                    [
                        "The \"good news\" is that the circus is in town. The \"bad news\" is that's right over Bugs Bunny's underground home. He wakes up as his place shakes like an earthquake hit it, when workers pound stakes into the ground and elephants stomp by, etc.<br /><br />To be more specific, the lions' cage is place exactly over Bugs' hole. The lion sniffs food, and by process of elimination, figures out it's a rabbit. Bugs, curious what all the racket is about, winds his way through the tunnel and winds up in the lion's mouth.<br /><br />I'll say for thing for BB: he is totally fearless, at least in this cartoon, and at least for 30 seconds. When he comes to his senses, he runs like crazy and we get a lion-versus-a rabbit battle the rest of the way. Once again, Bugs faces dumb opponent, one he calls \"Nero,\" but lion is fierce and Bugs will need all his wits and somewhat-fake bravado to fend off this beast.<br /><br />About half the gags are stupid and the other half funny, but always fast-moving, colorful and good enough to recommend. I mean, it's not everyday you can see a lion on a trapeze, or doing a hula dance!",
                        "Positive"
                    ],
                    [
                        "I never thought I would absolutly hate an Arnold Schwartzeneggar film, BUT this is is dreadful from the get go. there isnt one redeemable scene in the entire 123 long minutes. an absolute waste of time<br /><br /> thank yu<br /><br /> Jay harris",
                        "Negative"
                    ],
                    [
                        "Before I explain the \"Alias\" comment let me say that \"The Desert Trail\" is bad even by the standards of westerns staring The Three Stooges. In fact it features Carmen Laroux as semi- bad girl Juanita, when you hear her Mexican accent you will immediately recognize her as Senorita Rita from the classic Stooge short \"Saved by the Belle\". <br /><br />In \"The Desert Trail\" John Wayne gets to play the Moe Howard character and Eddy Chandler gets to play Curly Howard. Like their Stooge counterparts a running gag throughout the 53- minute movie is Moe hitting Curly. Wayne's character, a skirt chasing bully, is not very endearing, but is supposed to be the good guy. <br /><br />Playing a traveling rodeo cowboy Wayne holds up the rodeo box office at gunpoint and takes the prize money he would have won if the attendance proceeds had been good-the other riders have to settle for 25 cents on the dollar (actually even less after Wayne robs the box office). No explanation is given for Wayne's ripping off the riders and still being considered the hero who gets the girl. <br /><br />Things get complicated at this point because the villain (Al Ferguson) and his sidekick Larry Fine (played by Paul Fix-who would go on to play Sheriff Micah on television's \"The Rifleman\") see Wayne rob the box office and then steal the remainder of the money and kill the rodeo manager. Moe and Curly get blamed. <br /><br />So Moe and Curly move to another town to get away from the law and they change their names to Smith and Jones. Who do they meet first but their old friend Larry, whose sister becomes the 2nd half love interest (Senorita Rita is left behind it the old town and makes no further appearances in the movie). <br /><br />Larry's sister is nicely played by a radiantly beautiful Mary Kornman (now grown up but in her younger days she was one of the original cast members of Hal Roach's \"Our Gang\" shorts). Kornman is the main reason to watch the mega-lame western and her scenes with Moe and Curly are much better than any others in the production, as if they used an entirely different crew to film them. <br /><br />Even for 1935 the action sequences in this thing are extremely weak and the technical film- making is staggeringly bad. The two main chase scenes end with stock footage wide shots of a rider falling from a horse. Both times the editor cuts to a shot of one of the characters rolling on the ground, but there is no horse in the frame, the film stock is completely different, and the character has on different clothes than the stunt rider. There is liberal use of stock footage in other places, none of it even remotely convincing. <br /><br />One thing to watch for is a scene midway into the movie where Moe and Curly get on their horses and ride away (to screen right) from a cabin as the posse is galloping toward the cabin from the left. The cameraman follows the two stooges with a slow pan right and then does a whip pan to the left to reveal the approaching posse. Outside of home movies I have never seen anything like this, not because it is looks stupid (which it does) but because a competent director would never stage a scene in this manner. They would film the two riders leaving and then reposition the camera and film the posse approaching as a separate action. Or if they were feeling creative they would stage the sequence so the camera shows the riders in the foreground and the posse approaching in the background. <br /><br />Then again, what do I know? I'm only a child.",
                        "Negative"
                    ],
                    [
                        "i am rarely moved to make these kind of comments BUT after sitting through most of rankin's dreadful movie i feel like i have really earned the right to say what i feel about it! i couldn't actually make it right to the end, and became one of the half dozen or more walk outs (about 1/3rd of the audience) after the ragged plot, woeful dialogue and insulting characterisation became just too much to bear. this film is all pose and no art. all style and no substance. it is weighed down by dreadful acting, a genuinely dire script, indifferent cinematography and student-level production values. how it got funded, started, and finished is a mystery to me. i bet you a million quid it never goes on general release. the proper critics would tear it apart. a really bad film. shockingly bad. a really really really poor effort AND that is without even mentioning the gratuitous new-born-kitten-gets-dropped-into-a-deep-fat-fryer moment. totally meaningless, utterly lightweight, poorly put together; this movie is a dreadful embarrassment for uk cinema.",
                        "Negative"
                    ],
                    [
                        "I watched this movie for the hot guy--and even he sucked! He was the worst one--well, okay, I have to give props to that freaky police officer rapist guy too, he was even worse. The guy wasn't that cute in the end, he had the most terrible accent, and he was the most definite definition of hicksville idiot that can't stand up to his mom for the one he \"loves\" there's ever been. Overall, and if this makes any sense to you, when I go to pick up movies at the video store, I think to myself as I read the back of a movie that looks so/so, \"Well, at least it can't be worse than Carolina Moon.\" The most terrible movie, and the most terrible writing, acting, plot--everything in it made my gag reflexes want to do back flips. It was THE most horrid movie I will ever see, with Gabriela way up there too. I hated it, and trust me, if there was any number under 1 IMDb had for rating, I'd choose that in a heartbeat.",
                        "Negative"
                    ],
                    [
                        "Stargate SG-1 is a spin off of sorts from the 1994 movie \"Stargate.\" I am so glad that they decided to expand on the subject. The show gets it rolling from the very first episode, a retired Jack O'Neill has to go through the gate once more to meet with his old companion, Dr. Daniel Jackson. Through the first two episodes, we meet Samantha Carter, a very intelligent individual who lets no one walk over her, and there is Teal'c, a quiet, compassionate warrior who defies his false god and joins the team. <br /><br />The main bad guys are called the Gouald, they are parasites who can get inserted into one's brain, thus controlling them and doing evil deeds. Any Gouald who has a massive amount of power is often deemed as a \"System Lord.\" The warriors behind the Gouald are called Jaffa, who house the parasitic Gouald in their bodies until the Gouald can get inserted in a person's brain.<br /><br />Through the episodes, we mostly get to see SG-1, the exploratory team comprised of Jack/Daniel/Teal'c/and Sam, go through the wormhole that instantly transports them to other planets (this device is called the Stargate) and they encounter new cultures or bad guys. Some episodes are on-world, meaning that they do not go through the Stargate once in the episode and rather deal with pressing issues on Earth.<br /><br />Through the years, you start to see a decline in the SG-1 team as close knit, and more character-building story lines. This, in turn means even more on-world episodes, which is perfectly understandable.<br /><br />My rating: 8.75/10----While most of this show is good, there are some instances of story lines not always getting wrapped up and less of an emphasis on gate travel these last few years. But still, top notch science fiction!",
                        "Positive"
                    ]
                ]
            }
        }


class StandardEvalDataOutput(BaseModel):
    avg_loss: float
    avg_eval_f1_score: float

    class Config:
        schema_extra = {
            "example": {
                "avg_loss": 0.68117,
                "avg_eval_f1_score": 0.7
            }
        }


class PredictionOutputs(BaseModel):
    class_probs: List[List[float]]
    class_preds: List[Any]

    class Config:
        schema_extra = {
            "example": {
                "class_probs": [[0.48354560136795044, 0.5164543986320496], [0.48343217372894287, 0.5165678858757019],
                                [0.4839431643486023, 0.5160568356513977], [0.48435676097869873, 0.515643298625946],
                                [0.47995486855506897, 0.5200451612472534], [0.4862929880619049, 0.5137070417404175],
                                [0.4845595359802246, 0.5154405236244202], [0.4825523793697357, 0.5174476504325867],
                                [0.47728413343429565, 0.5227158069610596], [0.48190197348594666, 0.5180981159210205]],
                "class_preds": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
        }

class PredictApiParams(BaseModel):
    experiment_name: str
    dataset_name: str
    task: str
    train_dataset_size: int
    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim: Optional[int]
    custom_vocab_tokens: Optional[List[str]]
    hidden_dim: Optional[int]
    pre_train_training_size: Optional[int]
    eval_batch_size: Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "imdb_sa_nle_soft_match",
                "dataset_name": "imdb_sa_nle_soft_match",
                "train_dataset_size": 94,
                "task": "sa",
                "eval_batch_size": 50,
                "embeddings": "charngram.100d",
                "emb_dim": 100,
                "hidden_dim": 100,
                "pre_train_training_size": 50000
            }
        }


class PredictNextApiParams(BaseModel):
    params: PredictApiParams
    label_space: Dict[str, int]
    prediction_data: List[str]
    ner_label_space: Optional[List[str]]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "imdb_sa_nle_soft_match",
                    "dataset_name": "imdb_sa_nle_soft_match",
                    "train_dataset_size": 94,
                    "task": "sa",
                    "eval_batch_size": 50,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 100,
                    "pre_train_training_size": 50000
                },
                "label_space": {
                    "Positive": 0,
                    "Negative": 1
                },
                "prediction_data": [
                    "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie",
                    "Honestly, this is a very funny movie if you are looking for bad acting (Heather Graham could never live this down... it has three titles for a reason- to protect the guilty!), beautifully bad dialog (\"Do you like... ribs?\"), and a plot only a mother could approve, this is your Friday night entertainment! <br /><br />My roommate rented this under the title \"Terrified\" because he liked Heather Graham, but terrified is what we felt after the final credits. Not because the movie is scary, but because somebody actually paid money to make this turd on a movie reel.<br /><br />Horrible movie. There are a few no-name actors that provide some unintentional comedy, but nothing worth viewing. Heather Graham's dramatic climax also was one of the most pathetic and disturbing things I have ever witnessed. I award this movie no point, and may God have mercy on its soul.",
                    "This third Darkman was definitely better than the second one, but still far worse than the original movie. What made this one better than D2 was the fact that The Bad Guy had been changed and Durant was not brought back again. Furthermore there was actually some hint of character development when it came to the bad guy's family and Darkman himself. This made my heart soften and I gave this flick as much as 4/10, i.e. **/*****.",
                    "surely this film was hacked up by the studio? perhaps not but i feel there were serious flaws in the storytelling that if not attributed to the editing process could only be caused by grievously bad, criminal indeed, writing and directing.<br /><br />i understand the effect burton wished to achieve with the stylised acting similar to the gothic fairytale atmosphere of edward scissorhands, but here unfortunately it falls flat and achieves no mythical depth of tropes but only the offensive tripe of affectation. ie bad acting and shallow characterisation even for a fairytale.<br /><br />finally not that scary, indeed only mildly amusing in its attempts. the use of dialogue as a vehicle for plot background was clumsy and unnecessary. the mystery of who is the headless horseman would suffice, no need for the myth about a german mercenary, although christopher walken did cut a dashing figure but not that menacing - seeing the horsemans head makes him seem far friendlier that a decapitated inhuman nine foot tall spirit as in the original legend.<br /><br />no real rhythm or universal tone was ever established and not a classic in burtons oevure. stilted and clipped as my parting shot...",
                    "The \"good news\" is that the circus is in town. The \"bad news\" is that's right over Bugs Bunny's underground home. He wakes up as his place shakes like an earthquake hit it, when workers pound stakes into the ground and elephants stomp by, etc.<br /><br />To be more specific, the lions' cage is place exactly over Bugs' hole. The lion sniffs food, and by process of elimination, figures out it's a rabbit. Bugs, curious what all the racket is about, winds his way through the tunnel and winds up in the lion's mouth.<br /><br />I'll say for thing for BB: he is totally fearless, at least in this cartoon, and at least for 30 seconds. When he comes to his senses, he runs like crazy and we get a lion-versus-a rabbit battle the rest of the way. Once again, Bugs faces dumb opponent, one he calls \"Nero,\" but lion is fierce and Bugs will need all his wits and somewhat-fake bravado to fend off this beast.<br /><br />About half the gags are stupid and the other half funny, but always fast-moving, colorful and good enough to recommend. I mean, it's not everyday you can see a lion on a trapeze, or doing a hula dance!",
                    "I never thought I would absolutly hate an Arnold Schwartzeneggar film, BUT this is is dreadful from the get go. there isnt one redeemable scene in the entire 123 long minutes. an absolute waste of time<br /><br /> thank yu<br /><br /> Jay harris",
                    "Before I explain the \"Alias\" comment let me say that \"The Desert Trail\" is bad even by the standards of westerns staring The Three Stooges. In fact it features Carmen Laroux as semi- bad girl Juanita, when you hear her Mexican accent you will immediately recognize her as Senorita Rita from the classic Stooge short \"Saved by the Belle\". <br /><br />In \"The Desert Trail\" John Wayne gets to play the Moe Howard character and Eddy Chandler gets to play Curly Howard. Like their Stooge counterparts a running gag throughout the 53- minute movie is Moe hitting Curly. Wayne's character, a skirt chasing bully, is not very endearing, but is supposed to be the good guy. <br /><br />Playing a traveling rodeo cowboy Wayne holds up the rodeo box office at gunpoint and takes the prize money he would have won if the attendance proceeds had been good-the other riders have to settle for 25 cents on the dollar (actually even less after Wayne robs the box office). No explanation is given for Wayne's ripping off the riders and still being considered the hero who gets the girl. <br /><br />Things get complicated at this point because the villain (Al Ferguson) and his sidekick Larry Fine (played by Paul Fix-who would go on to play Sheriff Micah on television's \"The Rifleman\") see Wayne rob the box office and then steal the remainder of the money and kill the rodeo manager. Moe and Curly get blamed. <br /><br />So Moe and Curly move to another town to get away from the law and they change their names to Smith and Jones. Who do they meet first but their old friend Larry, whose sister becomes the 2nd half love interest (Senorita Rita is left behind it the old town and makes no further appearances in the movie). <br /><br />Larry's sister is nicely played by a radiantly beautiful Mary Kornman (now grown up but in her younger days she was one of the original cast members of Hal Roach's \"Our Gang\" shorts). Kornman is the main reason to watch the mega-lame western and her scenes with Moe and Curly are much better than any others in the production, as if they used an entirely different crew to film them. <br /><br />Even for 1935 the action sequences in this thing are extremely weak and the technical film- making is staggeringly bad. The two main chase scenes end with stock footage wide shots of a rider falling from a horse. Both times the editor cuts to a shot of one of the characters rolling on the ground, but there is no horse in the frame, the film stock is completely different, and the character has on different clothes than the stunt rider. There is liberal use of stock footage in other places, none of it even remotely convincing. <br /><br />One thing to watch for is a scene midway into the movie where Moe and Curly get on their horses and ride away (to screen right) from a cabin as the posse is galloping toward the cabin from the left. The cameraman follows the two stooges with a slow pan right and then does a whip pan to the left to reveal the approaching posse. Outside of home movies I have never seen anything like this, not because it is looks stupid (which it does) but because a competent director would never stage a scene in this manner. They would film the two riders leaving and then reposition the camera and film the posse approaching as a separate action. Or if they were feeling creative they would stage the sequence so the camera shows the riders in the foreground and the posse approaching in the background. <br /><br />Then again, what do I know? I'm only a child.",
                    "i am rarely moved to make these kind of comments BUT after sitting through most of rankin's dreadful movie i feel like i have really earned the right to say what i feel about it! i couldn't actually make it right to the end, and became one of the half dozen or more walk outs (about 1/3rd of the audience) after the ragged plot, woeful dialogue and insulting characterisation became just too much to bear. this film is all pose and no art. all style and no substance. it is weighed down by dreadful acting, a genuinely dire script, indifferent cinematography and student-level production values. how it got funded, started, and finished is a mystery to me. i bet you a million quid it never goes on general release. the proper critics would tear it apart. a really bad film. shockingly bad. a really really really poor effort AND that is without even mentioning the gratuitous new-born-kitten-gets-dropped-into-a-deep-fat-fryer moment. totally meaningless, utterly lightweight, poorly put together; this movie is a dreadful embarrassment for uk cinema.",
                    "I watched this movie for the hot guy--and even he sucked! He was the worst one--well, okay, I have to give props to that freaky police officer rapist guy too, he was even worse. The guy wasn't that cute in the end, he had the most terrible accent, and he was the most definite definition of hicksville idiot that can't stand up to his mom for the one he \"loves\" there's ever been. Overall, and if this makes any sense to you, when I go to pick up movies at the video store, I think to myself as I read the back of a movie that looks so/so, \"Well, at least it can't be worse than Carolina Moon.\" The most terrible movie, and the most terrible writing, acting, plot--everything in it made my gag reflexes want to do back flips. It was THE most horrid movie I will ever see, with Gabriela way up there too. I hated it, and trust me, if there was any number under 1 IMDb had for rating, I'd choose that in a heartbeat.",
                    "Stargate SG-1 is a spin off of sorts from the 1994 movie \"Stargate.\" I am so glad that they decided to expand on the subject. The show gets it rolling from the very first episode, a retired Jack O'Neill has to go through the gate once more to meet with his old companion, Dr. Daniel Jackson. Through the first two episodes, we meet Samantha Carter, a very intelligent individual who lets no one walk over her, and there is Teal'c, a quiet, compassionate warrior who defies his false god and joins the team. <br /><br />The main bad guys are called the Gouald, they are parasites who can get inserted into one's brain, thus controlling them and doing evil deeds. Any Gouald who has a massive amount of power is often deemed as a \"System Lord.\" The warriors behind the Gouald are called Jaffa, who house the parasitic Gouald in their bodies until the Gouald can get inserted in a person's brain.<br /><br />Through the episodes, we mostly get to see SG-1, the exploratory team comprised of Jack/Daniel/Teal'c/and Sam, go through the wormhole that instantly transports them to other planets (this device is called the Stargate) and they encounter new cultures or bad guys. Some episodes are on-world, meaning that they do not go through the Stargate once in the episode and rather deal with pressing issues on Earth.<br /><br />Through the years, you start to see a decline in the SG-1 team as close knit, and more character-building story lines. This, in turn means even more on-world episodes, which is perfectly understandable.<br /><br />My rating: 8.75/10----While most of this show is good, there are some instances of story lines not always getting wrapped up and less of an emphasis on gate travel these last few years. But still, top notch science fiction!"
                ]
            }
        }


class PredictStandardClfPayload(BaseModel):
    params: EvalStandardApiParams
    ner_label_space: Optional[List[str]]
    label_space: Dict[str, int]
    prediction_data: List[str]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "imdb_sa_nle_standard",
                    "dataset_name": "imdb_sa_nle_standard",
                    "task": "sa",
                    "eval_batch_size": 50,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 100
                },
                "label_space": {
                    "Positive": 0,
                    "Negative": 1
                },
                "prediction_data": [
                    "This film is underrated. I loved it. It was truly sweet and heartfelt. A family who struggles but isn't made into a dysfunctional family which is so typical of films today. The film didn't make it an issue that they have little money or are Dominican Republican the way Hollywood have.<br /><br />Instead the issue is Victor is immature and needs to grow up. He does, slowly, by the film's end. He has a ways to go, but it was a heartfelt attempt to move forward. His grandmother is very cute and the scene where the little boy throws up had me laughing for the longest time. A truly heartfelt indie",
                    "Honestly, this is a very funny movie if you are looking for bad acting (Heather Graham could never live this down... it has three titles for a reason- to protect the guilty!), beautifully bad dialog (\"Do you like... ribs?\"), and a plot only a mother could approve, this is your Friday night entertainment! <br /><br />My roommate rented this under the title \"Terrified\" because he liked Heather Graham, but terrified is what we felt after the final credits. Not because the movie is scary, but because somebody actually paid money to make this turd on a movie reel.<br /><br />Horrible movie. There are a few no-name actors that provide some unintentional comedy, but nothing worth viewing. Heather Graham's dramatic climax also was one of the most pathetic and disturbing things I have ever witnessed. I award this movie no point, and may God have mercy on its soul.",
                    "This third Darkman was definitely better than the second one, but still far worse than the original movie. What made this one better than D2 was the fact that The Bad Guy had been changed and Durant was not brought back again. Furthermore there was actually some hint of character development when it came to the bad guy's family and Darkman himself. This made my heart soften and I gave this flick as much as 4/10, i.e. **/*****.",
                    "surely this film was hacked up by the studio? perhaps not but i feel there were serious flaws in the storytelling that if not attributed to the editing process could only be caused by grievously bad, criminal indeed, writing and directing.<br /><br />i understand the effect burton wished to achieve with the stylised acting similar to the gothic fairytale atmosphere of edward scissorhands, but here unfortunately it falls flat and achieves no mythical depth of tropes but only the offensive tripe of affectation. ie bad acting and shallow characterisation even for a fairytale.<br /><br />finally not that scary, indeed only mildly amusing in its attempts. the use of dialogue as a vehicle for plot background was clumsy and unnecessary. the mystery of who is the headless horseman would suffice, no need for the myth about a german mercenary, although christopher walken did cut a dashing figure but not that menacing - seeing the horsemans head makes him seem far friendlier that a decapitated inhuman nine foot tall spirit as in the original legend.<br /><br />no real rhythm or universal tone was ever established and not a classic in burtons oevure. stilted and clipped as my parting shot...",
                    "The \"good news\" is that the circus is in town. The \"bad news\" is that's right over Bugs Bunny's underground home. He wakes up as his place shakes like an earthquake hit it, when workers pound stakes into the ground and elephants stomp by, etc.<br /><br />To be more specific, the lions' cage is place exactly over Bugs' hole. The lion sniffs food, and by process of elimination, figures out it's a rabbit. Bugs, curious what all the racket is about, winds his way through the tunnel and winds up in the lion's mouth.<br /><br />I'll say for thing for BB: he is totally fearless, at least in this cartoon, and at least for 30 seconds. When he comes to his senses, he runs like crazy and we get a lion-versus-a rabbit battle the rest of the way. Once again, Bugs faces dumb opponent, one he calls \"Nero,\" but lion is fierce and Bugs will need all his wits and somewhat-fake bravado to fend off this beast.<br /><br />About half the gags are stupid and the other half funny, but always fast-moving, colorful and good enough to recommend. I mean, it's not everyday you can see a lion on a trapeze, or doing a hula dance!",
                    "I never thought I would absolutly hate an Arnold Schwartzeneggar film, BUT this is is dreadful from the get go. there isnt one redeemable scene in the entire 123 long minutes. an absolute waste of time<br /><br /> thank yu<br /><br /> Jay harris",
                    "Before I explain the \"Alias\" comment let me say that \"The Desert Trail\" is bad even by the standards of westerns staring The Three Stooges. In fact it features Carmen Laroux as semi- bad girl Juanita, when you hear her Mexican accent you will immediately recognize her as Senorita Rita from the classic Stooge short \"Saved by the Belle\". <br /><br />In \"The Desert Trail\" John Wayne gets to play the Moe Howard character and Eddy Chandler gets to play Curly Howard. Like their Stooge counterparts a running gag throughout the 53- minute movie is Moe hitting Curly. Wayne's character, a skirt chasing bully, is not very endearing, but is supposed to be the good guy. <br /><br />Playing a traveling rodeo cowboy Wayne holds up the rodeo box office at gunpoint and takes the prize money he would have won if the attendance proceeds had been good-the other riders have to settle for 25 cents on the dollar (actually even less after Wayne robs the box office). No explanation is given for Wayne's ripping off the riders and still being considered the hero who gets the girl. <br /><br />Things get complicated at this point because the villain (Al Ferguson) and his sidekick Larry Fine (played by Paul Fix-who would go on to play Sheriff Micah on television's \"The Rifleman\") see Wayne rob the box office and then steal the remainder of the money and kill the rodeo manager. Moe and Curly get blamed. <br /><br />So Moe and Curly move to another town to get away from the law and they change their names to Smith and Jones. Who do they meet first but their old friend Larry, whose sister becomes the 2nd half love interest (Senorita Rita is left behind it the old town and makes no further appearances in the movie). <br /><br />Larry's sister is nicely played by a radiantly beautiful Mary Kornman (now grown up but in her younger days she was one of the original cast members of Hal Roach's \"Our Gang\" shorts). Kornman is the main reason to watch the mega-lame western and her scenes with Moe and Curly are much better than any others in the production, as if they used an entirely different crew to film them. <br /><br />Even for 1935 the action sequences in this thing are extremely weak and the technical film- making is staggeringly bad. The two main chase scenes end with stock footage wide shots of a rider falling from a horse. Both times the editor cuts to a shot of one of the characters rolling on the ground, but there is no horse in the frame, the film stock is completely different, and the character has on different clothes than the stunt rider. There is liberal use of stock footage in other places, none of it even remotely convincing. <br /><br />One thing to watch for is a scene midway into the movie where Moe and Curly get on their horses and ride away (to screen right) from a cabin as the posse is galloping toward the cabin from the left. The cameraman follows the two stooges with a slow pan right and then does a whip pan to the left to reveal the approaching posse. Outside of home movies I have never seen anything like this, not because it is looks stupid (which it does) but because a competent director would never stage a scene in this manner. They would film the two riders leaving and then reposition the camera and film the posse approaching as a separate action. Or if they were feeling creative they would stage the sequence so the camera shows the riders in the foreground and the posse approaching in the background. <br /><br />Then again, what do I know? I'm only a child.",
                    "i am rarely moved to make these kind of comments BUT after sitting through most of rankin's dreadful movie i feel like i have really earned the right to say what i feel about it! i couldn't actually make it right to the end, and became one of the half dozen or more walk outs (about 1/3rd of the audience) after the ragged plot, woeful dialogue and insulting characterisation became just too much to bear. this film is all pose and no art. all style and no substance. it is weighed down by dreadful acting, a genuinely dire script, indifferent cinematography and student-level production values. how it got funded, started, and finished is a mystery to me. i bet you a million quid it never goes on general release. the proper critics would tear it apart. a really bad film. shockingly bad. a really really really poor effort AND that is without even mentioning the gratuitous new-born-kitten-gets-dropped-into-a-deep-fat-fryer moment. totally meaningless, utterly lightweight, poorly put together; this movie is a dreadful embarrassment for uk cinema.",
                    "I watched this movie for the hot guy--and even he sucked! He was the worst one--well, okay, I have to give props to that freaky police officer rapist guy too, he was even worse. The guy wasn't that cute in the end, he had the most terrible accent, and he was the most definite definition of hicksville idiot that can't stand up to his mom for the one he \"loves\" there's ever been. Overall, and if this makes any sense to you, when I go to pick up movies at the video store, I think to myself as I read the back of a movie that looks so/so, \"Well, at least it can't be worse than Carolina Moon.\" The most terrible movie, and the most terrible writing, acting, plot--everything in it made my gag reflexes want to do back flips. It was THE most horrid movie I will ever see, with Gabriela way up there too. I hated it, and trust me, if there was any number under 1 IMDb had for rating, I'd choose that in a heartbeat.",
                    "Stargate SG-1 is a spin off of sorts from the 1994 movie \"Stargate.\" I am so glad that they decided to expand on the subject. The show gets it rolling from the very first episode, a retired Jack O'Neill has to go through the gate once more to meet with his old companion, Dr. Daniel Jackson. Through the first two episodes, we meet Samantha Carter, a very intelligent individual who lets no one walk over her, and there is Teal'c, a quiet, compassionate warrior who defies his false god and joins the team. <br /><br />The main bad guys are called the Gouald, they are parasites who can get inserted into one's brain, thus controlling them and doing evil deeds. Any Gouald who has a massive amount of power is often deemed as a \"System Lord.\" The warriors behind the Gouald are called Jaffa, who house the parasitic Gouald in their bodies until the Gouald can get inserted in a person's brain.<br /><br />Through the episodes, we mostly get to see SG-1, the exploratory team comprised of Jack/Daniel/Teal'c/and Sam, go through the wormhole that instantly transports them to other planets (this device is called the Stargate) and they encounter new cultures or bad guys. Some episodes are on-world, meaning that they do not go through the Stargate once in the episode and rather deal with pressing issues on Earth.<br /><br />Through the years, you start to see a decline in the SG-1 team as close knit, and more character-building story lines. This, in turn means even more on-world episodes, which is perfectly understandable.<br /><br />My rating: 8.75/10----While most of this show is good, there are some instances of story lines not always getting wrapped up and less of an emphasis on gate travel these last few years. But still, top notch science fiction!"
                ]
            }
        }


class StandardNERTrainingApiParams(BaseModel):
    experiment_name: str
    dataset_name: str
    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim: Optional[int]
    content_emb: Optional[Literal['none', 'elmo']]
    seed: Optional[int]
    digit2zero: Optional[bool]
    hidden_dim: Optional[int]
    dropout: Optional[float]
    use_char_rnn: Optional[bool]
    use_crf_layer: Optional[bool]
    optimizer: Optional[Literal['sgd', 'adam']]
    trig_optimizer: Optional[Literal['sgd', 'adam']]
    learning_rate: Optional[float]
    momentum: Optional[float]
    l2: Optional[float]
    num_epochs: Optional[int]
    pre_train_num_epochs: Optional[int]
    batch_size: Optional[int]
    lr_decay: Optional[float]
    build_data: bool


class StandardNERTrainingPayload(BaseModel):
    params: StandardNERTrainingApiParams
    labeled_data: List[LabeledDoc]
    dev_data: Optional[List[LabeledDoc]]


class LeanLifeStandardNERParams(BaseModel):
    experiment_name: str
    dataset_name: str
    project_type: str
    project_id: int

    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                 'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                 'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]

    emb_dim: Optional[int]
    # content_emb: Optional[Literal['none', 'elmo']]
    seed: Optional[int]
    digit2zero: Optional[bool]
    hidden_dim: Optional[int]
    dropout: Optional[float]
    use_char_rnn: Optional[bool]
    use_crf_layer: Optional[bool]
    # optimizer: Optional[Literal['sgd', 'adam']]
    # trig_optimizer: Optional[Literal['sgd', 'adam']]
    learning_rate: Optional[float]
    momentum: Optional[float]
    l2: Optional[float]
    num_epochs: Optional[int]
    pre_train_num_epochs: Optional[int]
    batch_size: Optional[int]
    lr_decay: Optional[float]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "test_experiment_1",
                "dataset_name": "test_dataset",
                "project_type": "Named Entity Recognition"
            }
        }


class LeanLifeStandardNERPayload(BaseModel):
    lean_life_data: LeanLifeStandardData
    params: LeanLifeStandardNERParams


class StandardNEREvalDataOutput(BaseModel):
    precision: float
    recall: float
    f1: float

    class Config:
        schema_extra = {
            "example": {
                "precision": 0.68117,
                "recall": 0.7,
                "f1": 0.6,
            }
        }


class EvalStandardNERApiParams(BaseModel):
    experiment_name: str
    dataset_name: str
    task: Literal['sa', 're', 'ner']

    batch_size: Optional[int]
    embeddings: Optional[Literal['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d',
                                  'glove.840B.300d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d',
                                  'glove.twitter.27B.200d', 'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']]
    emb_dim: Optional[int]
    hidden_dim: Optional[int]
    seed: Optional[int]

    class Config:
        schema_extra = {
            "example": {
                "experiment_name": "conll_ner_standard",
                "dataset_name": "conll_ner_standard",
                "task": "ner",
                "eval_batch_size": 10,
                "embeddings": "charngram.100d",
                "emb_dim": 100,
                "hidden_dim": 100
            }
        }


class EvalStandardNERPayload(BaseModel):
    params: EvalStandardNERApiParams
    eval_data: List[Tuple[str, str]]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "conll_ner_standard",
                    "dataset_name": "conll_ner_standard",
                    "task": "ner",
                    "eval_batch_size": 10,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 200
                },
                "eval_data": [
                    [
                        "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
                        "O O B-LOC O O O O B-PER O O O O"
                    ],
                    [
                        "Nadim Ladki",
                        "B-PER I-PER"
                    ]
                ]
            }
        }


class NERPredictionOutputs(BaseModel):
    class_preds: List[Any]

    class Config:
        schema_extra = {
            "example": {
                "class_preds": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            }
        }


class PredictStandardNERPayload(BaseModel):
    params: EvalStandardNERApiParams
    prediction_data: List[str]

    class Config:
        schema_extra = {
            "example": {
                "params": {
                    "experiment_name": "conll_ner_standard",
                    "dataset_name": "conll",
                    "task": "ner",
                    "eval_batch_size": 10,
                    "embeddings": "charngram.100d",
                    "emb_dim": 100,
                    "hidden_dim": 100
                },
                "prediction_data": [
                    "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
                    "AL-AIN , United Arab Emirates 1996-12-06"
                ]
            }
        }
