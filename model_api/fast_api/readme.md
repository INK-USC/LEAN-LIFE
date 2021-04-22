1. `sh create_data_dirs.sh`
2. `pip install -r rqs.txt`
3. modify nltk source code:  in ```nltk/parse/chart.py```,  line 680, modify function ```parse```, change ```for edge in self.select(start=0, end=self._num_leaves,lhs=root):```  to  ```for edge in self.select(start=0, end=self._num_leaves):```
4. `python -m spacy download en_core_web_sm`
5. `uvicorn main:app --reload --port=9000`
6. http://127.0.0.1:9000/docs (for interactive docs)

Model Training Code will require CUDA 10.1, and if you're on linux you'll have to install it via
the pytorch site. The torch version is 1.7.0. For more help here, check [here](https://pytorch.org/get-started/previous-versions/#v170)

Additionally to set the GPUs available for model training, please change line 26 in `main.py`
