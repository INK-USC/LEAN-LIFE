1. `sh create_data_dirs.sh`
2. `pip install -r rqs.txt`
3. `python -m spacy download en_core_web_sm`
4. place tacred_train.json, tared_dev.json, tacred_test.json into `data` folder
5. `cd training`
6. `python pre_train_find_module.py --build_data` (defaults achieve 90% f1)
7. `python train_bilstm_tacred.py --build_data --experiment_name="insertAnyTextHere"`
    * there are several available params you can set from the command line
    * builds data for both strict and soft labeling, only uses strict data
    * data pre-processing will take sometime, due to tuning of parser.
    * Note: match_batch_size == batch_size in the bilstm case
8. (Optional) `python train_next_classifier_tacred.py --experiment_name="insertAnyTextHere"`
    * --build_data flag is needed if you didn't run step 5

For both step 4 and 5, subsequent trials on the same dataset don't need the "--build_data" flag; data that has already been computed does not need to be computed again, it is stored to disk.

Directory Descriptions:

*CCG* : everything to do with parsing of explanations and creation of strict and soft labeling functions
* main file: CCG/parser.py

*models* : model files

*tests* : test files, tests are a good place to understand a lot of the functions. train_util_functions.py doesn't have tests around it yet though. To run a test, run the following: `pytest test_file_name.py`. Example: `pytest ccg_util_test.py`

*training* : all code to do with training of models