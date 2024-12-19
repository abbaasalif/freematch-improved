# FreeMatch-Improved

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

This repository has the implementation of FreeMatch and some of the modifications that I tried to implement. The following commits will discuss the changes and are works in progress. If you would like to use the original losses described in the paper, you need to go to the utils folder and change the __init__.py file to get the appropriate losses imported. This is not optimal but will be revised in the future versions.

This work is entirely inspired by the official implementation of the paper and also from https://github.com/shreejalt/freematch-pytorch


---

## 🚀 Features

- **Enhanced FreeMatch Algorithm**: Improved upon the original FreeMatch approach.
- **Semi-Supervised Learning**: Balances labeled and unlabeled data effectively.
- **Customizable**: Easily adjust hyperparameters for experimentation.

---

## 📖 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)
- [Citations](#citations)

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abbaasalif/freematch-improved.git
   cd freematch-improved
   ```
2. Install Anaconda or Miniconda for your platform (Linux/Windows/Mac). This repository is platform-agnostic.
3. Create the conda environment using the provided `environment.yml`:
   ```bash
   conda env create -f environment.yml
   ```
   *Why use Anaconda?* It provides pre-built CUDA runtime libraries, so all you need are the appropriate drivers for your GPU to get started.
   ```

---

## 🎯 Usage

### Configuration

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

### Running the Model
1. Train the model using labeled and unlabeled datasets:
   ```bash
   python3 main.py --config-file config/cifar10/freematch_cifar10_10.yaml
   ```

2. Evaluate the model:
   ```bash
   python3 main.py --validate-only --config-file config/cifar10/freematch_cifar10_10.yaml --resume logs/freematch_cifar10_10/model_ckpt/best_checkpoint.pth.
   ```

### Configurations
Customize the `configs/config.yaml` file to:
- Change datasets.
- Adjust hyperparameters (e.g., learning rate, batch size).

---

## 🖼️ Examples

### Training Example
```bash
python train.py --config configs/cifar10.yaml
```

### Visualizing Results
The training process logs metrics like accuracy and loss. Visualize them with TensorBoard:
```bash
tensorboard --logdir=logs/
```

---


## 🤝 Contributing

We welcome contributions! Here’s how you can help:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---
## 📜 Citations

If you find this repository helpful, please consider citing the following works:

### FreeMatch
```
@article{wang2023freematch,
  title={FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning},
  author={Wang, Yidong and Chen, Hao and Heng, Qiang and Hou, Wenxin and Fan, Yue and Wu, Zhen and Wang, Jindong and Savvides, Marios and Shinozaki, Takahiro and Raj, Bhiksha and Schiele, Bernt and Xie, Xing},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

### USB: Unified Semi-supervised Learning Benchmark
```
@inproceedings{usb2022,
  doi = {10.48550/ARXIV.2208.07204},
  url = {https://arxiv.org/abs/2208.07204},
  author = {Wang, Yidong and Chen, Hao and Fan, Yue and Sun, Wang and Tao, Ran and Hou, Wenxin and Wang, Renjie and Yang, Linyi and Zhou, Zhi and Guo, Lan-Zhe and Qi, Heli and Wu, Zhen and Li, Yu-Feng and Nakamura, Satoshi and Ye, Wei and Savvides, Marios and Raj, Bhiksha and Shinozaki, Takahiro and Schiele, Bernt and Wang, Jindong and Xie, Xing and Zhang, Yue},
  title = {USB: A Unified Semi-supervised Learning Benchmark for Classification},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
  year = {2022}
}
```

### Domain Generalization
```
@article{zhou2022domain,
  title={Domain generalization: A survey},
  author={Zhou, Kaiyang and Liu, Ziwei and Qiao, Yu and Xiang, Tao and Loy, Chen Change},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```

### Domain Adaptive Ensemble Learning
```
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

---



## 🛠️ Future Work

- Integrating advanced loss functions.
- Adding support for larger datasets like ImageNet.
- Enhancing model interpretability.

---

## 🙌 Acknowledgements

- [FreeMatch Paper](https://arxiv.org/abs/2205.07246)
- Contributors and the open-source community.

---
