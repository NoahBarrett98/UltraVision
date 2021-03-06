export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
hparam_search --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/Ultra/Images \
               --data_name FetalPlanes \
               --model_name DenseNet169 \
               --use_scheduler False \
               --batch_size 32 \
               --num_epochs  2 \
               --val_size 0.1 \
               --optimizer_name Adam \
               --one_channel True \
               --use_og_split False \
               --normalize_option "paper"