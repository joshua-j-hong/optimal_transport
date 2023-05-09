export WORKLOC=/home/hong
DATA_FILE=$WORKLOC/data/pre_processed_data/awesomeAlign/EnFr/enfr.src-tgt
ref_align=$WORKLOC/data/pre_processed_data/awesomeAlign/EnFr/enfr.gold
reftype='--oneRef'
#MODEL_NAME_OR_PATH=bert-base-multilingual-cased
MODEL_NAME_OR_PATH=$WORKLOC/awesome-align-models/supervised/
OUTPUT_FILE=$WORKLOC/awesome-align/output.temp
CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 32 \

python $WORKLOC/awesome-align/tools/aer.py ${ref_align} $OUTPUT_FILE --fAlpha 0.5 $reftype