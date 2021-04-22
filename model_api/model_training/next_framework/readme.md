Adaptation of code from here: https://github.com/INK-USC/NExT

**Installation Instructions**
1. `sh create_data_dirs.sh`
2. `pip install -r rqs.txt`
    * If on Linux, you must download pytorch with CUDA 10.1 compatibility
    * For more instructions, check [here](https://pytorch.org/get-started/previous-versions/#v170)
3. `python -m spacy download en_core_web_sm`
4. modify nltk source code:  in ```nltk/parse/chart.py```,  line 680, modify function ```parse```, change ```for edge in self.select(start=0, end=self._num_leaves,lhs=root):```  to  ```for edge in self.select(start=0, end=self._num_leaves):```
5. place TACRED's train.json, dev.json, test.json into `data` folder
6. run through `Prepare Tacred Data.ipynb` to prepare TACRED data
7. `cd training`
8. `python run_scripts/pre_train_find_module.py --build_data`* (defaults achieve 90% f1)
9. `python run_scripts/train_next_classifier.py --build_data --experiment_name="insertAnyTextHere"`* (defaults achieve 42.4% f1)
    * there are several available params you can set from the command line
    * builds data for both strict and soft labeling, only uses strict data
    * data pre-processing will take sometime, due to tuning of parser.
    * Note: match_batch_size == batch_size in the bilstm case

\* paths will break here, so you will have to update path info if you want to run these scripts. Alternatively you can check out this [repo](https://github.com/Rahul-Khanna/NExT)

For both step 8 and 9, subsequent trials on the same dataset don't need the "--build_data" flag; data that has already been computed does not need to be computed again, it is stored to disk.

Directory Descriptions:

*CCG_new* : everything to do with parsing of explanations and creation of strict and soft labeling functions
* main file: CCG_new/parser.py

*models* : model files

*tests* : test files, tests are a good place to understand a lot of the functions. train_util_functions.py doesn't have tests around it yet though. To run a test, run the following: `pytest test_file_name.py`. Example: `pytest ccg_util_test.py`. Some tests might not pass due to data not existing, read comments in those tests to understand how to build needed data.

*training* : all code to do with training of models