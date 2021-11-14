CUDA_SET_VISIBLE_DEVICES=1
train_model --label_dir /home/noah/data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv \
               --data_dir /home/noah/data/FETAL_PLANES_ZENODO/Images \
               --data_name FetalPlanes \
               --model HighResCNN \
               --train_strategy train_classification \
               --use_scheduler True \
               --batch_size 32 \
               --num_epochs  10 \
               --lr 0.0002 \
               --momentum 0.9