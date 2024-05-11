#! /usr/bin
# Change n, key

#nhop="2"
#nkey="two"
#model="kgsf"
save_path="/root/autodl-fs/${model}/inspired-conceptnet"

file1="concept_subkg.txt"
file2="word2id.json"
file1_alias="conceptnet_kg_${nkey}_hop.txt"
file2_alias="conceptnet_${nkey}_hop_word2id.json"

data_path="$(pwd)/data/dataset/inspired/nltk"
kg_path="$(pwd)/ali-kg/conceptnet-inspired/${nhop}-hop"
tmp_path="$(pwd)/data"

# move and rename n-hop files
mv "${data_path}/${file1}" "${tmp_path}"
mv "${data_path}/${file2}" "${tmp_path}"
cp "${kg_path}/${file1_alias}" "${data_path}"
mv "${data_path}/${file1_alias}" "${data_path}/${file1}"
cp "${kg_path}/${file2_alias}" "${data_path}"
mv "${data_path}/${file2_alias}" "${data_path}/${file2}"

# run model
python run_crslab.py --config "config/crs/${model}/inspired.yaml"

# backup
mkdir -p "${save_path}/${nhop}-hop/"
cp -R log "${save_path}/${nhop}-hop/"
zip -r "inspired-conceptnet-${nhop}hop-model.zip" model
mv "inspired-conceptnet-${nhop}hop-model.zip" "${save_path}/${nhop}-hop/"

# clean up
sh scripts/clean.sh
rm "${data_path}/${file1}"
rm "${data_path}/${file2}"
mv "${tmp_path}/${file1}" "${data_path}"
mv "${tmp_path}/${file2}" "${data_path}"
