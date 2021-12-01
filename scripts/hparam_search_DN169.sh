export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
hparam_search --label_dir /home/noah/data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/FETAL_PLANES_ZENODO/Images \
               --data_name FetalPlanes2 \
               --model_name DenseNet169 \
               --use_scheduler False \
               --batch_size 32 \
               --num_epochs  15 \
               --val_size 0.1 \
               --optimizer_name SGD \
               --one_channel False
