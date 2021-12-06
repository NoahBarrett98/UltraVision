import os
from setuptools import setup

setup(name='UltraVision',
      version='0.1',
      description='Self-Supervision for Ultrasound Imaging',
      author='Connor Taylor, Keelan Earle, Noah Barrett',
      include_package_data=True,
      zip_safe=False,
      install_requires=[
        'click==7.1.2',
        "click_logging",
        'efficientnet-pytorch',
        "matplotlib",
        'numpy',
        'opencv-python==4.5.2.52',
        'optuna==2.7.0',
        'pandas==1.1.3',
        "Pillow",
        'protobuf==3.15.8',
        'scikit-image==0.17.2',
        'scikit-learn==0.24.2',
        "scipy",
        'tensorboard==2.3.0',
        'tensorboardX==2.1',
        'torch==1.8.1',
        "seaborn",
        "torchvision",
        'tqdm==4.50.2',
        'mlflow'
      ],
      entry_points={
          'console_scripts': [
                'train_model = UltraVision.main:train_model',
                'train_baselines = UltraVision.main:runSciLearn',
                'train_bootstrap = UltraVision.main:train_bootstrap',
                'hparam_search = UltraVision.main:hparam_search_wrapper',
                'evaluate_tuned_models = UltraVision.evaluation:evaluate_tuned_models',
                'evaluate_model = UltraVision.evaluation:evaluate_model'
          ]
      }
)
