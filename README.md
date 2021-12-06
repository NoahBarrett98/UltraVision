# UltraVision
CS 6505 Group Project 

## installation
 
This project is set up as a python package to install

```
conda create --name UltraVision python=3.9
cd path/to/UltraVision
pip install -e .
conda install protobuf
```

## running experiment

Example of running experiment: train simcl
```
train_bootstrap --label_dir data/FETAL_PLANES_DB_data.csv \
               --data_dir data/Images \
               --data_name FetalPlanes \
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
               --save_results_dir UltraVision/experiments/transfer_self_supervised_to_classification \
               --num_bootstraps 5 \
               --freeze_base False
```

## Reproducibility of Results

All experiments discussed in report can be found in the UltraVision/experiments directory. The default seeds in the code 
are what were used in the experiments. 


## Tracking Experiments

This project uses both tensorboard and mlflow to track experiments. In order to track experiments with tensorboard, 
one must set --use_tensorboard to True, mlflow is automatically logged. All training sessions are logged to the working
directory.

### using tensorboard
```
tensorboard --logdir path/to/tensorboard/example_20211028_09:32
```
### using mlflow
```
mlflow ui
```
