# Bilevel Transfer Attack (BETAK)

This repo contains the source code for the paper named "Advancing Generalized Transfer Attack with Initialization Derived Bilevel Optimization and Dynamic Sequence Truncation", which has been accepted to IJCAI 2024.

## Environment
- Python
- PyTorch
- Higher
- timm

## Datasets(ImageNet)

Following [BPA](https://github.com/Trustworthy-AI-Group/BPA), we also randomly sample 5,000 images pertaining to the 1,000 categories
from ILSVRC 2012 validation set, which could be classified correctly by all the victim models. The corresponding CSV file are saved as **`data/imagenet/selected_imagenet_resnet50.csv`** and **`data/imagenet/selected_imagenet_vgg19_bn.csv`**.

Before running, you should download the ILSVRC 2012 validation set and specify the directory using parameters `--imagenet_val_dir`. Your directory structure is as follows:

```
imagenet-val
    ├── n07880968
    |   ├── ILSVRC2012_val_00011685.JPEG
    |   └── ...
    ├── n02927161
    |   ├── ILSVRC2012_val_00011685.JPEG
    |   └── ...
    └── ...
```
## Models

The pretrained models can be downloaded at [here](https://drive.google.com/drive/folders/1wyQRUc2Jmyi7ZZB_p_iTJ1vaqf18fDWq?usp=sharing), then extract them to `ckpt/`.

## Attack & Test with BETAK

Untargeted attack using **BETAK**+ **BPA** + **PGD** as an example:

```bash
python attack_eval_imagenet_betak.py --epsilon 0.03 --sgm_lambda 1.0 --niters 10 --method max_relu_silu_pgd --batch_size 25 --save_dir data/betak_imagenet/max_relu_silu_pgd_attack_resnet50_meta_incv3meta_lr_2.0_meta_step3_inner_loop10 --device_id 0 --imagenet_val_dir imagenet/val --model_name resnet50 --alpha 0.006 --inner_loop 10 --attack_lr 2.0 --meta_steps 3
```
You should modify the default directory for the validation set according to your system path.

## Citation

