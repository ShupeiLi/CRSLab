#! /usr/bin
# Change n, model

nhop="2"
model="kbrd"
save_path="/root/autodl-fs/${model}/redial-dbpedia"

file1="dbpedia_subkg.json"
file2="entity2id.json"
file3="movie_ids.json"
data_path="$(pwd)/data/dataset/redial/nltk"
kg_path="$(pwd)/ali-kg/dbpedia-redial/${nhop}-hop"
tmp_path="$(pwd)/data"

# move n-hop files
mv "${data_path}/${file1}" "${tmp_path}"
mv "${data_path}/${file2}" "${tmp_path}"
mv "${data_path}/${file3}" "${tmp_path}"
cp "${kg_path}/${file1}" "${data_path}"
cp "${kg_path}/${file2}" "${data_path}"
cp "${kg_path}/${file3}" "${data_path}"

# run model
python run_crslab.py --config "config/crs/${model}/redial.yaml"

# backup
mkdir -p "${save_path}/${nhop}-hop/"
cp -R log "${save_path}/${nhop}-hop/"
zip -r "redial-dbpedia-${nhop}hop-model.zip" model
mv "redial-dbpedia-${nhop}hop-model.zip" "${save_path}/${nhop}-hop/"

# clean up
sh scripts/clean.sh
rm "${data_path}/${file1}"
rm "${data_path}/${file2}"
rm "${data_path}/${file3}"
mv "${tmp_path}/${file1}" "${data_path}"
mv "${tmp_path}/${file2}" "${data_path}"
mv "${tmp_path}/${file3}" "${data_path}"
