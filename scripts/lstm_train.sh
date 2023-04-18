# trains the bi-lstm-crf model
python -m bi_lstm_crf ../data/preprocessed/ner/ie \
--model_dir ../data/results/ner \
--max_seq_len 1024 \
--num_epoch 30