#!/bin/sh


export WorkLOC=/home/hong/optimal_transport

#yours

DATASET=ALIGN6/dev_data
#DATASET=JaEn_training/dev_data
SOURCE=dev.src
TARGET=dev.tgt

datadir=$WorkLOC/data/pre_processed_data/accAlign/$DATASET/

EVAL_GOLD_FILE=$datadir/dev.talp

EXTRACTION=unbalancedOT
FERTILITY_DISTRIBUTION=l2_norm
COST_FUNCTION=cosine_sim
ENTROPY_REGULARIZATION=0.1

SRC=$WorkLOC/data/pre_processed_data/accAlign/$DATASET/$SOURCE
TGT=$WorkLOC/data/pre_processed_data/accAlign/$DATASET/$TARGET

OUTPUT_DIR=$WorkLOC/AccAlign/infer_output
Model=sentence-transformers/LaBSE
ADAPTER=$WorkLOC/accalign-models/supervised/checkpoint-1200/
ADAPTER=$WorkLOC/AccAlign/checkpoint-adapter

#Model=$WorkLOC/accalign-models/uot_supervised_no_adapter_uniform/


# python $WorkLOC/AccAlign/train_alignment_adapter.py \
#    --infer_path $OUTPUT_DIR \
#    --model_name_or_path $Model \
#    --extraction $EXTRACTION \
#    --infer_data_file_src $SRC \
#    --infer_data_file_tgt $TGT \
#    --per_gpu_train_batch_size 40 \
#    --gradient_accumulation_steps 1 \
#    --align_layer 6 \
#    --fertility_distribution $FERTILITY_DISTRIBUTION \
#    --cost_function $COST_FUNCTION \
#    --eval_gold_file $EVAL_GOLD_FILE \
#    --do_param_search \
#    --entropy_regularization $ENTROPY_REGULARIZATION \
#    --gold_one_index \

python $WorkLOC/AccAlign/train_alignment_adapter.py \
   --infer_path $OUTPUT_DIR \
   --adapter_path $ADAPTER \
   --model_name_or_path $Model \
   --extraction $EXTRACTION \
   --infer_data_file_src $SRC \
   --infer_data_file_tgt $TGT \
   --per_gpu_train_batch_size 40 \
   --gradient_accumulation_steps 1 \
   --align_layer 6 \
   --fertility_distribution $FERTILITY_DISTRIBUTION \
   --cost_function $COST_FUNCTION \
   --eval_gold_file $EVAL_GOLD_FILE \
   --do_param_search \
   --entropy_regularization $ENTROPY_REGULARIZATION \
   --gold_one_index \

exit