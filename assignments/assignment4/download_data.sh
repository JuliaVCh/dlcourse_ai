mkdir data
cd data
wget --no-check-certificate https://storage.googleapis.com/dlcourse_ai/train.zip
wget --no-check-certificate https://storage.googleapis.com/dlcourse_ai/test.zip
unzip -q "train.zip"
unzip -q "test.zip"

