export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
train_model --label_dir /home/noah/data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/FETAL_PLANES_ZENODO/Images \
               --data_name FetalPlanes \
               --model_name ResNet18 \
               --train_strategy train_classification \
               --use_scheduler True \
               --batch_size 1 \
               --num_epochs  1 \
               --optimizer_name SGD \
               --lr 0.0002 \
               --momentum 0.9 \
               --exp_name testing \
               --use_tensorboard True