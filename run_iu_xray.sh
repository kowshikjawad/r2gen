ANN_PATH=${1:-"data/iu_xray/annotation_layman.json"}

python main.py \
--image_dir data/iu_xray/images/ \
--ann_path "$ANN_PATH" \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 16 \
--epochs 100 \
--save_dir results/iu_xray \
--step_size 50 \
--gamma 0.1 \
--seed 9223

