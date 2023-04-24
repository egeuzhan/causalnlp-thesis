# causalnlp-thesis


### Instructions

1) folder data should be placed in the root directory, results should be placed into data. install_dep.sh can be used to install all necessary packages. <br>

2) To run the model, run predict.sh  <br>

3) To change the train and test files, use atlop_predict.sh, docunet_predict.sh. <br>


To test the RE models with the original dataset, please set the test_file to test_ie.json in the prediction scripts, and lines 50, 53 in src/results.py 

To show the results of the predictions in RE, remove the comments in models/atlop/train.py line 232 and models/docunet/train_balanceloss.py line 328. <br>

final predictions can be found in predictions/final_predictions.json. The predictions will be put into the respective folders in predictions/ if a single model is used <br>






### References
---
##### Implementations of models modified from:
ATLOP: https://github.com/wzhouad/ATLOP <br>
DocuNet: https://github.com/zjunlp/DocuNet <br>
Bi-LSTM-CRF: https://github.com/jidasheng/bi-lstm-crf <br>

CREST: https://github.com/phosseini/CREST <br>
