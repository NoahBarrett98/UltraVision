export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
train_bootstrap --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/Ultra/Images \
               --data_name FetalPlanes2 \
               --model_name DenseNet169 \
               --train_strategy train_classification \
               --use_scheduler False \
               --batch_size 32 \
               --train_strategy train_classification \
               --num_epochs  10 \
               --val_size 0.1 \
               --lr 0.00828149 \
               --optimizer_name SGD \
               --one_channel False \
               --save_results_dir /home/noah/UltraVision/experiments/linear_classification/imnet \
               --num_bootstraps 5 \
               --freeze_base True