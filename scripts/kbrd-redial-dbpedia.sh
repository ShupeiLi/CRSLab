#! /usr/bin
# Change n
#mv ./data/dataset/redial/nltk/dbpedia_subkg.json ./data &&
#mv ./data/dataset/redial/nltk/entity2id.json ./data &&
#mv ./data/dataset/redial/nltk/movie_ids.json ./data &&
#cp ./ali-kg/dbpedia-redial/2-hop/dbpedia_subkg.json ./data/dataset/redial/nltk/ &&
#cp ./ali-kg/dbpedia-redial/2-hop/entity2id.json ./data/dataset/redial/nltk/ &&
#cp ./ali-kg/dbpedia-redial/2-hop/movie_ids.json ./data/dataset/redial/nltk/ &&
python run_crslab.py --config config/crs/kbrd/redial.yaml --save_system
