# FreeMatch - Self-adaptive Thresholding for Semi-supervised Learning

This repository has the implementation of FreeMatch and some of the modifications that I tried to implement. The modifications will be discussed more in the following commits and are work in progress. If you would like to use the original losses as described in the paper you need to go to the utils folder and then change the __init__.py file to get the appropriate losses imported. This is not optimal but will be revised in the future version of the updates that we will propose in the code. 

This work is entirely inspired by the official implementation of the paper and also from https://github.com/shreejalt/freematch-pytorch

## Disclaimer

All the theorems, propositions, and the proof are taken from the paper by Wang et.al. I have just reproduced the paper to show the main experiments and the results following the propositions of their work in Semi-Supervised Learning. I would like to thank the authors for their outstanding work on a new approach to semi-supervised learning and detailed analysis of the working of the same. To get into the details of all the loss functions and their proofs, read the original paper.

## Running the Experiments

### Setup

1. `git clone https://github.com/abbaasalif/freematch-improvements`
2. `cd freematch-improvements`
3. `Download and install anaconda for your distribution - This code is platform agnostic and we want it work on Linux and Windows Machines`
4. `conda env create -f environment.yml`
We highly recommend using anaconda or miniconda as they have channels that will download all required cuda runtime libraries into your machine to use GPU. All you need to do is install suitable drivers and get started.
### Running the scripts
These config files are identical to the https://github.com/shreejalt/freematch-pytorch repository. 
All the config files for CIFAR10 and CIFAR100 are present in the `config` folder. It follows the `yacs` and logging format inspired from [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). You can visit the given link to learn more about `yacs.config.CfgNode` structure. 

The script `main.py` contains argument parser which can be used to overwrite the config file of the experiment. 

```python
	main.py [-h] [--config-file CONFIG_FILE] [--run-name RUN_NAME]
               [--output-dir OUTPUT_DIR] [--log-dir LOG_DIR] [--tb-dir TB_DIR]
               [--resume-checkpoint RESUME_CHECKPOINT] [--cont-train]
               [--validate-only] [--train-batch-size TRAIN_BATCH_SIZE]
               [--test-batch-size TEST_BATCH_SIZE] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE
                        Path to the config file of the experiment
  --run-name RUN_NAME   Run name of the experiment
  --output-dir OUTPUT_DIR
                        Directory to save model checkpoints
  --log-dir LOG_DIR     Directory to save the logs
  --tb-dir TB_DIR       Directory to save tensorboard logs
  --resume-checkpoint RESUME_CHECKPOINT
                        Resume path of the checkpoint
  --cont-train          Flag to continue training
  --validate-only       Flag for validation only
  --train-batch-size TRAIN_BATCH_SIZE
                        Training batch size
  --test-batch-size TEST_BATCH_SIZE
                        Testing batch size
  --seed SEED           Seed
```

To execute the training, execute the command 

`python3 main.py --config-file config/cifar10/freematch_cifar10_10.yaml`

This will start the training by running the `train()` function in `trainer.py`. 



You can also use your model checkpoints to validate the results. Run the command <br\>

`python3 main.py --validate-only --config-file config/cifar10/freematch_cifar10_10.yaml --resume logs/freematch_cifar10_10/model_ckpt/best_checkpoint.pth`. 



Note that, you need to add `--validate-only` flag everytime you want to test your model. This file will run the `test()` function from `tester.py` file. 
## Citations

```
@article{wang2023freematch,
  title={FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning},
  author={Wang, Yidong and Chen, Hao and Heng, Qiang and Hou, Wenxin and Fan, Yue and and Wu, Zhen and Wang, Jindong and Savvides, Marios and Shinozaki, Takahiro and Raj, Bhiksha and Schiele, Bernt and Xie, Xing},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
@inproceedings{usb2022,
  doi = {10.48550/ARXIV.2208.07204},
  url = {https://arxiv.org/abs/2208.07204},
  author = {Wang, Yidong and Chen, Hao and Fan, Yue and Sun, Wang and Tao, Ran and Hou, Wenxin and Wang, Renjie and Yang, Linyi and Zhou, Zhi and Guo, Lan-Zhe and Qi, Heli and Wu, Zhen and Li, Yu-Feng and Nakamura, Satoshi and Ye, Wei and Savvides, Marios and Raj, Bhiksha and Shinozaki, Takahiro and Schiele, Bernt and Wang, Jindong and Xie, Xing and Zhang, Yue},
  title = {USB: A Unified Semi-supervised Learning Benchmark for Classification},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
  year = {2022}
}

@article{zhou2022domain,
  title={Domain generalization: A survey},
  author={Zhou, Kaiyang and Liu, Ziwei and Qiao, Yu and Xiang, Tao and Loy, Chen Change},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@article{zhou2021domain,
  title={Domain adaptive ensemble learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={8008--8018},
  year={2021},
  publisher={IEEE}
}
```

