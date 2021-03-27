1. `sh create_data_dirs.sh`
2. `pip install -r rqs.txt`
3. `python -m spacy download en_core_web_sm`
4. `uvicorn main:app --reload --port=9000`
5. http://127.0.0.1:9000/docs (for interactive docs)