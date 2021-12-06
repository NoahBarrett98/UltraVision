export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
train_bootstrap --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/Ultra/Images \
               --data_name FetalPlanes2 \
               --model_name DenseNet169 \
               --train_strategy train_classification \
               --use_scheduler False \
               --batch_size 32 \
               --train_strategy train_classification \
               --num_epochs  15 \
               --val_size 0.1 \
               --lr 0.00828149 \
               --optimizer_name SGD \
               --one_channel False \
               --save_results_dir /home/noah/UltraVision/experiments/linear_classification/simclr_rerun \
               --load_model_from /home/noah/AI_IWK/CUS/notebooks/experiments/expn19_simclr_fp/long_train_retrain/900.pt \
               --num_outputs_pretrained 128 \
               --num_bootstraps 5 \
               --freeze_base True

