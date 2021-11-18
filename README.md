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

Example of running experiment
```
train_model --label_dir /data/FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv \
               --data_dir /data/FETAL_PLANES_ZENODO/Images \
               --data_name FetalPlanes \
               --model ResNet18 \
               --train_strategy train_classification \
               --use_scheduler True \
               --batch_size 1 \
               --num_epochs  1 \
               --optimizer_name SGD \
               --lr 0.0002 \
               --momentum 0.9 \
               --exp_name testing \
               --use_tensorboard True
```

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
