## Modified from the original script run_bert.sh of ATLOP
python ../models/atlop/train.py --data_dir ../data/preprocessed/re/ie/set1 \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--train_file train_annotated.json \
--dev_file val.json \
--test_file test_ie_ent.json \
--train_batch_size 32 \
--test_batch_size 16 \
--gradient_accumulation_steps 1 \
--num_labels 1 \
--learning_rate 5e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 10 \
--seed 66 \
--num_class 2 \
--load_path ../data/results/atlop/ie/atlop_model.json
