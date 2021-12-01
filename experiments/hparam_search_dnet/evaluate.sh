export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
evaluate_tuned_models --label_dir /home/noah/data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv \
                     --data_dir /home/noah/data/FETAL_PLANES_ZENODO/Images \
                     --data_name FetalPlanes2 \
                     --model_name DenseNet169 \
                      --val_size 0.1 \
                      --checkpoint_dir /home/noah/ray_results/train_func_2021-11-30_13-52-26 \
                      --checkpoint_num 14 \
                      --save_results /home/noah/UltraVision/experiments/hparam_search_dnet/random_split