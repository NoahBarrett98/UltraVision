export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
hparam_search --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/Ultra/Images \
               --data_name FetalPlanes \
               --model_name DenseNet169 \
               --use_scheduler False \
               --batch_size 32 \
               --num_epochs  5 \
               --val_size 0.1 \
               --optimizer_name SGD \
               --one_channel False \
               --use_og_split False \
               --normalize_option "custom" \
               --num_trials 15 \
               --save_output_dir /home/noah/UltraVision/experiments/hparam_search_dnet/random_split