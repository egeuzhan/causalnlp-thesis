cd ../src
chmod x ner.py
python ner.py 
cd ../scripts
chmod x atlop_predict.sh
./atlop_predict.sh
chmod x docunet_predict.sh
./docunet_predict.sh
cd ../src
chmod x results.py
python results.py --model atlop
cd ../scripts


