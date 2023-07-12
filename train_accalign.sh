#!/bin/sh


WorkLOC=/home/hong/optimal_transport #yours




TRAIN_FILE_SRC=$WorkLOC/data/pre_processed_data/accAlign/ALIGN6/train_data/train.src
TRAIN_FILE_TGT=$WorkLOC/data/pre_processed_data/accAlign/ALIGN6/train_data/train.tgt
TRAIN_FILE_ALIGN=$WorkLOC/data/pre_processed_data/accAlign/ALIGN6/train_data/train.talp

# TRAIN_FILE_SRC=$WorkLOC/data/pre_processed_data/accAlign/JaEn_training/train_data/train.src
# TRAIN_FILE_TGT=$WorkLOC/data/pre_processed_data/accAlign/JaEn_training/train_data/train.tgt
# TRAIN_FILE_ALIGN=$WorkLOC/data/pre_processed_data/accAlign/JaEn_training/train_data/train.talp

#TRAIN_FILE_SRC=$WorkLOC/xxx/train.src
#TRAIN_FILE_TGT=$WorkLOC/xxx/train.tgt
#TRAIN_FILE_ALIGN=$WorkLOC/xxx/train.talp

EVAL_FILE_SRC=$WorkLOC/data/pre_processed_data/accAlign/ALIGN6/dev_data/dev.src
EVAL_FILE_TGT=$WorkLOC/data/pre_processed_data/accAlign/ALIGN6/dev_data/dev.tgt
Eval_gold_file=$WorkLOC/data/pre_processed_data/accAlign/ALIGN6/dev_data/dev.talp

# EVAL_FILE_SRC=$WorkLOC/data/pre_processed_data/accAlign/JaEn_training/dev_data/dev.src
# EVAL_FILE_TGT=$WorkLOC/data/pre_processed_data/accAlign/JaEn_training/dev_data/dev.tgt
# Eval_gold_file=$WorkLOC/data/pre_processed_data/accAlign/JaEn_training/dev_data/dev.talp

#EVAL_FILE_SRC=$WorkLOC/xxx/dev.src
#EVAL_FILE_TGT=$WorkLOC/xxx/dev.tgt
#Eval_gold_file=$WorkLOC/xxx/dev.talp

OUTPUT_DIR_ADAPTER=$WorkLOC/accalign-models/uot_supervised_no_adapter_uniform
Model=sentence-transformers/LaBSE

EXTRACTION=unbalancedOT
ALIGNMENT_THRESHOLD=0.26
ENTROPY_REGULARIZATION=0.1
MARGINAL_REGULARIZATION=0.06
FERTILITY_DISTRIBUTION=l2_norm
COST_FUNCTION=cosine_sim

#OUTPUT_DIR_ADAPTER=$WorkLOC/adapter_output
#Model=$WorkLOC/models/LaBSE

EVAL_RES=$WorkLOC/AccAlign/eval_result

# learning rate 1e-4 max steps 1200
CUDA_VISIBLE_DEVICES=0 python $WorkLOC/AccAlign/train_alignment_adapter.py \
    --output_dir_adapter $OUTPUT_DIR_ADAPTER \
    --eval_res_dir $EVAL_RES \
    --model_name_or_path $Model \
    --extraction $EXTRACTION \
    --train_so \
    --do_train \
    --do_eval \
    --train_data_file_src $TRAIN_FILE_SRC \
    --train_data_file_tgt $TRAIN_FILE_TGT \
    --eval_data_file_src $EVAL_FILE_SRC \
    --eval_data_file_tgt $EVAL_FILE_TGT \
    --per_gpu_train_batch_size 40 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 20 \
    --learning_rate 1e-5 \
    --save_steps 100  \
    --max_steps 1200 \
    --align_layer 6 \
    --logging_steps 50 \
    --eval_gold_file $Eval_gold_file \
    --alignment_threshold $ALIGNMENT_THRESHOLD \
    --entropy_regularization $ENTROPY_REGULARIZATION \
    --marginal_regularization $MARGINAL_REGULARIZATION \
    --fertility_distribution $FERTILITY_DISTRIBUTION \
    --train_gold_file $TRAIN_FILE_ALIGN \
    --cost_function $COST_FUNCTION \
    --overwrite_output_dir \
    --gold_one_index \
    --toggle_adapter 
    #--output_dir $OUTPUT_DIR \
    #--max_steps 1200 \

exit

