export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
train_model --label_dir /home/noah/data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/FETAL_PLANES_ZENODO/Images \
               --data_name FetalPlanes2 \
               --model_name DenseNet169 \
               --train_strategy train_classification \
               --use_scheduler True \
               --batch_size 32 \
               --num_epochs  15 \
               --optimizer_name SGD \
               --lr 0.00977542  \
               --momentum 0.9 \
               --exp_name reproduce_dnet \
               --tensorboard_name /home/noah/UltraVision/experiments/reproduce_dnet \
               --one_channel False