# 3D DMFNet for Real-time Brain Tumor Segmentation

This repository is the work of "3D Dilated Multi-Fiber Network for Real-time Brain Tumor Segmentation in MRI". You could click the link to access the [paper](https://arxiv.org/pdf/1904.03355.pdf). The multimodal brain tumor dataset (BraTS 2018) could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html).


## Dilated multi-fiber network

![Image text](https://github.com/China-LiuXiaopeng/BraTS-DMFNet/blob/master/fig/Architecture.jpg)

## Requirements
* python 3.6
* pytorch 0.4 or 1.0
* nibabel
* pickle 

## Implementation

Download the BraTS2018 dataset and change the path in ./experiments/PATH.yaml.

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

(Optional) You could split the training set into k-fold for the **cross-validation** experiment by means of the sklearn lib.

```
python split.py
```

### Training

Sync bacth normalization is used in our networks, a proper batch size is recommended, i.e., batch_size=8. Multi-gpu training is necessary to obtain a decent result.
```
python train_all.py --gpu=0,1,2,3 --cfg=DMFNet_GDL_all --batch_size=8
```

### Test

**Table. Dice scores for the enhancing tumor, whole tumor and tumor core respevtively** 

| Model         | Params (M) | FLOPs (G) | Dice_ET (%) | Dice_WT (%) | Dice_TC (%) | 
| :-------------|:----------:|:----------:|:-----------:|:-----------:|:-----------:|
| 0.75x MFNet   | 1.81 | 13.36 | 79.34 | 90.22 | 84.25 | 
| MFNet         | 3.19 | 20.61 | 79.91 | 90.43 | 84.61 | 
| DMFNet        | 3.88 | 27.04 | 80.12 | 90.62 | 84.54 |

We have provided the trained weights for download (Baidu drive). Please download the [weight](https://pan.baidu.com/s/1dRyo9ZvisZvAwO4TVen2Pg) (提取码: t8xu) in the ./ckpt/DMFNet_GDL_all/here, then run the testing code, you could obtaion hte results as paper reported.

```
python test.py --mode=0 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=DMFNet_GDL_all --gpu=0
```

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```
@inproceedings{chen2019dmfnet,
  title={3D Dilated Multi-Fiber Network for Real-time Brain Tumor Segmentation in MRI},
  author={Chen, Chen and Liu, Xiaopeng and Ding, Meng and Zheng, Junfeng and Li, Jiangyun},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2019}
}
```

## Acknowledge

1. [MFNet](https://github.com/cypw/PyTorch-MFNet)
2. [BraTS2018-tumor-segmentation](https://github.com/ieee820/BraTS2018-tumor-segmentation)
3. [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
