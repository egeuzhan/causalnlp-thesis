cd ../src
chmod u+x ner.py
python ner.py 
cd ../scripts
chmod u+x atlop_predict.sh
./atlop_predict.sh
chmod u+x docunet_predict.sh
./docunet_predict.sh
cd ../src
chmod u+x results.py
python results.py --model atlop
cd ../scripts


