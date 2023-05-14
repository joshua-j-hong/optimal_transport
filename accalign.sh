#!/bin/sh


export WorkLOC=/home/hong/optimal_transport
#yours

SRC=$WorkLOC/data/pre_processed_data/accAlign/DeEn/de
TGT=$WorkLOC/data/pre_processed_data/accAlign/DeEn/en

#SRC=$WorkLOC/xxx/roen/roen.src
#TGT=$WorkLOC/xxx/roen/roen.tgt

OUTPUT_DIR=$WorkLOC/AccAlign/infer_output
ADAPTER=$WorkLOC/AccAlign/checkpoint-adapter
#ADAPTER=$WorkLOC/accalign-models/supervised/checkpoint-1200/
Model=sentence-transformers/LaBSE

#OUTPUT_DIR=$WorkLOC/xxx/infer_output
#ADAPTER=$WorkLOC/xxx/adapter
#Model=$WorkLOC/xxx/LaBSE

# python $WorkLOC/AccAlign/train_alignment_adapter.py \
#     --infer_path $OUTPUT_DIR \
#     --model_name_or_path $Model \
#     --extraction 'softmax' \
#     --infer_data_file_src $SRC \
#     --infer_data_file_tgt $TGT \
#     --per_gpu_train_batch_size 40 \
#     --gradient_accumulation_steps 1 \
#     --align_layer 6 \
#     --softmax_threshold 0.1 \
#     --do_test \

python $WorkLOC/AccAlign/train_alignment_adapter.py \
   --infer_path $OUTPUT_DIR \
   --adapter_path $ADAPTER \
   --model_name_or_path $Model \
   --extraction 'softmax' \
   --infer_data_file_src $SRC \
   --infer_data_file_tgt $TGT \
   --per_gpu_train_batch_size 40 \
   --gradient_accumulation_steps 1 \
   --align_layer 6 \
   --softmax_threshold 0.1 \
   --do_test \

datadir=$WorkLOC/data/pre_processed_data/accAlign/DeEn/
ref_align=$datadir/alignmentDeEn.talp
reftype='--oneRef'


for LayerNum in `seq 1 12`; do
    echo "=====AER shifted for layer=${LayerNum}..."
    python $WorkLOC/AccAlign/aer.py ${ref_align} $WorkLOC/AccAlign/infer_output/XX2XX.align.$LayerNum --fAlpha 0.5 $reftype
done

exit