export CUDA_VISIBLE_DEVICES=GPU_ID


DATA_ITEMS=("DATA_ITEM_NAME")
WM_ITEMS=("WM_ITEM_NAME")

for i in "${!DATA_ITEMS[@]}"; do
    data_item="${DATA_ITEMS[$i]}"
    wm_item="${WM_ITEMS[$i]}"

    DATA_PATH="/path/to/data/$data_item"
    LOG_PATH="/path/to/output/${data_item}_${wm_item}.log"

    MODEL_PATH1="/path/to/output/${data_item}"
    MODEL_PATH2="/path/to/output/${data_item}"

    python train.py \
        -s $DATA_PATH \
        --port 6016 \
        --model_path $MODEL_PATH1 \
        --iterations 10000 \
        --resolution 4 \
        --white_background \
        --detector_lr 0.0001 \
        --eval

    python render.py -m $MODEL_PATH1
    python metrics.py -m $MODEL_PATH1

    python train_stega.py \
        -s $DATA_PATH \
        --start_checkpoint ${MODEL_PATH1}/chkpnt10000.pth \
        --watermark_path /path/to/wm/${wm_item}.png \
        --model_path $MODEL_PATH2 \
        --port 6018 \
        --resolution 4 \
        --iterations 20000 \
        --target_weight 1.0 \
        --white_background \
        --detector_lr 0.0001 \
        --eval

    python render_stega.py \
        -m $MODEL_PATH2 \
        --watermark_path /path/to/wm/${wm_item}.png \
        --iteration 20000 \
        --skip_train

    python metrics.py \
        -m $MODEL_PATH2

done

