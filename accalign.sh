#!/bin/sh


export WorkLOC=/home/hong/optimal_transport
#yours

DATASET=ALIGN6/dev_data
# DATASET=JaEn_training/dev_data
SOURCE=dev.src
TARGET=dev.tgt

DATASET=ZhEn
SOURCE=zh
TARGET=en

EXTRACTION=unbalancedOT
ALIGNMENT_THRESHOLD=0.34
ENTROPY_REGULARIZATION=0.1
MARGINAL_REGULARIZATION=0.14
MASS_TRANSPORTED=1
FERTILITY_DISTRIBUTION=l2_norm
COST_FUNCTION=cosine_sim

SRC=$WorkLOC/data/pre_processed_data/accAlign/$DATASET/$SOURCE
TGT=$WorkLOC/data/pre_processed_data/accAlign/$DATASET/$TARGET

#SRC=$WorkLOC/xxx/roen/roen.src
#TGT=$WorkLOC/xxx/roen/roen.tgt

OUTPUT_DIR=$WorkLOC/AccAlign/infer_output
ADAPTER=$WorkLOC/AccAlign/checkpoint-adapter
#ADAPTER=$WorkLOC/accalign-models/supervised/checkpoint-1200/
Model=sentence-transformers/LaBSE

#Model=$WorkLOC/accalign-models/uot_supervised_no_adapter/

#OUTPUT_DIR=$WorkLOC/xxx/infer_output
#ADAPTER=$WorkLOC/xxx/adapter
#Model=$WorkLOC/xxx/LaBSE

# python $WorkLOC/AccAlign/train_alignment_adapter.py \
#    --infer_path $OUTPUT_DIR \
#    --model_name_or_path $Model \
#    --extraction $EXTRACTION \
#    --infer_data_file_src $SRC \
#    --infer_data_file_tgt $TGT \
#    --per_gpu_train_batch_size 40 \
#    --gradient_accumulation_steps 1 \
#    --align_layer 6 \
#    --alignment_threshold $ALIGNMENT_THRESHOLD \
#    --entropy_regularization $ENTROPY_REGULARIZATION \
#    --marginal_regularization $MARGINAL_REGULARIZATION \
#    --mass_transported $MASS_TRANSPORTED \
#    --fertility_distribution $FERTILITY_DISTRIBUTION \
#    --cost_function $COST_FUNCTION \
#    --do_test \

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
   --alignment_threshold $ALIGNMENT_THRESHOLD \
   --entropy_regularization $ENTROPY_REGULARIZATION \
   --marginal_regularization $MARGINAL_REGULARIZATION \
   --fertility_distribution $FERTILITY_DISTRIBUTION \
   --cost_function $COST_FUNCTION \
   --do_test \

datadir=$WorkLOC/data/pre_processed_data/accAlign/$DATASET/
ref_align=$datadir/alignment$DATASET.talp
#ref_align=$datadir/dev.talp
reftype='--oneRef'


for LayerNum in `seq 1 12`; do
    echo "=====AER shifted for layer=${LayerNum}..."
    python $WorkLOC/AccAlign/aer.py ${ref_align} $WorkLOC/AccAlign/infer_output/XX2XX.align.$LayerNum --fAlpha 0.5 $reftype
    #python $WorkLOC/AccAlign/aer.py ${ref_align} $WorkLOC/AccAlign/infer_output/XX2XX.align.$LayerNum --fAlpha 0.5 $reftype --most_common_errors 5 --source $SRC --target $TGT
done

exit