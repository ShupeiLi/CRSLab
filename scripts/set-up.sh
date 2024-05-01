#! /usr/bin
git pull origin main
cp -R ~/autodl-fs/ali-kg .
cp ~/autodl-fs/embedding.zip ./data
unzip ./data/embedding.zip -d ./data
rm ./data/embedding.zip
