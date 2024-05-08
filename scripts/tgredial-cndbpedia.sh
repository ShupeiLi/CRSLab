#! /usr/bin
# Change n, key, model

#nhop="2"
#nkey="two"
#model="kbrd"
save_path="/root/autodl-fs/${model}/tgredial-cndbpedia"

file1="entity2id.json"
file2="cn-dbpedia.txt"
file3="movie_ids.json"
file1_alias="cndbpedia-${nkey}-hop-kg-entity2id.json"
file2_alias="cndbpedia-${nkey}-hop-kg.txt"
file3_alias="${nkey}-hop-movie_ids.json"

data_path="$(pwd)/data/dataset/tgredial/pkuseg"
kg_path="$(pwd)/ali-kg/cndbpedia/${nhop}-hop"
tmp_path="$(pwd)/data"

# move and rename n-hop files
mv "${data_path}/${file1}" "${tmp_path}"
mv "${data_path}/${file2}" "${tmp_path}"
mv "${data_path}/${file3}" "${tmp_path}"
cp "${kg_path}/${file1_alias}" "${data_path}"
mv "${data_path}/${file1_alias}" "${data_path}/${file1}"
cp "${kg_path}/${file2_alias}" "${data_path}"
mv "${data_path}/${file2_alias}" "${data_path}/${file2}"
cp "${kg_path}/${file3_alias}" "${data_path}"
mv "${data_path}/${file3_alias}" "${data_path}/${file3}"

# run model
python run_crslab.py --config "config/crs/${model}/tgredial.yaml"

# backup
mkdir -p "${save_path}/${nhop}-hop/"
cp -R log "${save_path}/${nhop}-hop/"
zip -r "tgredial-cndbpedia-${nhop}hop-model.zip" model
mv "tgredial-cndbpedia-${nhop}hop-model.zip" "${save_path}/${nhop}-hop/"

# clean up
sh scripts/clean.sh
rm "${data_path}/${file1}"
rm "${data_path}/${file2}"
rm "${data_path}/${file3}"
mv "${tmp_path}/${file1}" "${data_path}"
mv "${tmp_path}/${file2}" "${data_path}"
mv "${tmp_path}/${file3}" "${data_path}"
