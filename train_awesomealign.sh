TRAIN_FILE=/home/hong/data/pre_processed_data/awesomeAlign/ALIGN6/train_data/train.src-tgt
TRAIN_GOLD_FILE=/home/hong/data/pre_processed_data/awesomeAlign/ALIGN6/train_data/train.gold
EVAL_FILE=/home/hong/data/pre_processed_data/awesomeAlign/ALIGN6/dev_data/dev.src-tgt
EVAL_GOLD_FILE=/home/hong/data/pre_processed_data/awesomeAlign/ALIGN6/dev_data/dev.gold
#OUTPUT_DIR=/home/hong/awesome-align-models/supervised
OUTPUT_DIR=/home/hong/awesome-align-models/self-supervised

#Supervised
# CUDA_VISIBLE_DEVICES=0 awesome-train \
#     --output_dir=$OUTPUT_DIR \
#     --model_name_or_path=bert-base-multilingual-cased \
#     --extraction 'softmax' \
#     --do_train \
#     --train_mlm \
#     --train_tlm \
#     --train_tlm_full \
#     --train_so \
#     --train_psi \
#     --train_data_file=$TRAIN_FILE \
#     --train_gold_file=$TRAIN_GOLD_FILE \
#     --per_gpu_train_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 5 \
#     --learning_rate 2e-5 \
#     --save_steps 10000 \
#     --max_steps 40000 \
#     --do_eval \
#     --eval_data_file=$EVAL_FILE \
#     --eval_gold_file=$EVAL_GOLD_FILE \
#     --gold_one_index

#Unsupervised
CUDA_VISIBLE_DEVICES=0 awesome-train \
    --output_dir=$OUTPUT_DIR \
    --model_name_or_path=bert-base-multilingual-cased \
    --extraction 'softmax' \
    --do_train \
    --train_mlm \
    --train_tlm \
    --train_tlm_full \
    --train_so \
    --train_psi \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --save_steps 10000 \
    --max_steps 40000 \
    --do_eval \
    --eval_data_file=$EVAL_FILE \