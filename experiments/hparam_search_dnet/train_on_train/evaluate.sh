export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# custom #
evaluate_model --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/Ultra/Images \
                     --data_name FetalPlanes \
                     --model_name DenseNet169 \
                      --val_size 0.1 \
                      --checkpoint /home/noah/UltraVision/experiments/hparam_search_dnet/train_on_train/models/model_custom.pt \
                      --save_results /home/noah/UltraVision/experiments/hparam_search_dnet/train_on_train \
                      --one_channel False \
                      --use_og_split True \
                      --normalize_option "custom"
## paper #
#evaluate_model --label_dir /home/noah/data/Ultra/FETAL_PLANES_DB_data.csv \
#               --data_dir /home/noah/data/Ultra/Images \
#               --data_name FetalPlanes \
#               --model_name DenseNet169 \
#                --val_size 0.1 \
#                --checkpoint /home/noah/UltraVision/experiments/hparam_search_dnet/train_on_train/models/train_on_train_paper.pt \
#                --save_results /home/noah/UltraVision/experiments/hparam_search_dnet/train_on_train \
#                --one_channel True \
#                --use_og_split True \
#                --normalize_option "paper"