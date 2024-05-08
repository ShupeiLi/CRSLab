#! /usr/bin
# Change n, key

nhop="2"
nkey="two"
model="kgsf"
save_path="/root/autodl-fs/${model}/redial-conceptnet"

file1="conceptnet_subkg.txt"
file2="concept2id.json"
file1_alias="conceptnet_${nkey}_hop_edges.txt"
file2_alias="conceptnet_${nkey}_hop_key2index.json"

data_path="$(pwd)/data/dataset/redial/nltk"
kg_path="$(pwd)/ali-kg/conceptnet-redial/${nhop}-hop"
tmp_path="$(pwd)/data"

# move n-hop files
mv "${data_path}/${file1}" "${tmp_path}"
mv "${data_path}/${file2}" "${tmp_path}"
cp "${kg_path}/${file1_alias}" "${data_path}"
mv "${data_path}/${file1_alias}" "${data_path}/${file1}"
cp "${kg_path}/${file2_alias}" "${data_path}"
mv "${data_path}/${file2_alias}" "${data_path}/${file2}"

# run model
python run_crslab.py --config "config/crs/${model}/redial.yaml"

# backup
mkdir -p "${save_path}/${nhop}-hop/"
cp -R log "${save_path}/${nhop}-hop/"
zip -r "redial-conceptnet-${nhop}hop-model.zip" model
mv "redial-conceptnet-${nhop}hop-model.zip" "${save_path}/${nhop}-hop/"

# clean up
sh scripts/clean.sh
rm "${data_path}/${file1}"
rm "${data_path}/${file2}"
mv "${tmp_path}/${file1}" "${data_path}"
mv "${tmp_path}/${file2}" "${data_path}"
