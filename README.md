# causalnlp-thesis


### Instructions

1) Install the dependencies on a conda environment through install_dep.sh <br>

2) Folder data should be placed in the root directory, results should be placed into data. <br>

3) To run the model, run predict.sh  <br>

- Training the models can be perfomed through the atlop_train.sh, docunet_train.sh and lstm_train.sh files. <br>

- To change the train and test files, set the train_file, dev_file and test_file parameters in atlop_predict.sh and docunet_predict.sh. <br>

- To test the RE models with the original dataset, please set the test_file to test_ie.json in the prediction scripts, and lines 50, 53 in src/results.py <br>

- For displaying the evaluation metrics with the original test_ie.json dataset, remove the comments in models/atlop/train.py line 232 and models/docunet/train_balanceloss.py line 328. <br>

- Final predictions can be found in predictions/final_predictions.json. The predictions will be put into the respective folders in predictions/ if a model is used separately. <br>






### References
---
##### Implementations of models modified from:
ATLOP: https://github.com/wzhouad/ATLOP <br>
DocuNet: https://github.com/zjunlp/DocuNet <br>
Bi-LSTM-CRF: https://github.com/jidasheng/bi-lstm-crf <br>

CREST: https://github.com/phosseini/CREST <br>
