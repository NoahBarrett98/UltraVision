export CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
train_model \
        --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
        --data_dir /home/noah/data/Ultra/Images \
        --dataset_name FetalPlanes \
        --validation_split 0.1 \
        --one_channel False \
        --num_epochs 1000 \
        --milestone 200 \
        --lr 0.0002\
        --momentum 0.9 \
        --batch_size 256 \
        --model_type DenseNet169 \
        --tensorboard_name  \
        --seed 10 \
        --data_split_seed 10 \
        --save_model_dir  \
        --optimizer_name SGD \
        --pretrained True \
        --sample_strategy None \
        --train_strategy train_simclr