## Modified from the original script run_docred.sh of DocuNet

#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1 

# -------------------Training Shell Script--------------------
if true; then
  transformer_type=bert
  channel_type=context-based
  if [[ $transformer_type == bert ]]; then
    bs=16
    bl=5e-5
    uls=(5e-5)
    accum=1
    for ul in ${uls[@]}
    do
    python -u ../models/docunet/train_balanceloss.py --data_dir ../data/preprocessed/re/ie/set1 \
    --channel_type $channel_type \
    --bert_lr $bl \
    --transformer_type $transformer_type \
    --model_name_or_path bert-base-cased \
    --train_file train_annotated.json \
    --dev_file val.json \
    --test_file test_ie_ent.json \
    --train_batch_size $bs \
    --test_batch_size $bs \
    --gradient_accumulation_steps $accum \
    --num_labels 1 \
    --learning_rate $ul \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.06 \
    --num_train_epochs 10 \
    --seed 66 \
    --num_class 2 \
    --save_path ../data/results/docunet/ie/docunet_model.json \
    #--log_dir ../../resultsnewcrestonesent32/docunetset5/train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.log
    #--save_path ./1train_bert-lr${bl}_accum${accum}_unet-lr${ul}_type_${channel_type}.pt \
    
    done
  fi
fi
