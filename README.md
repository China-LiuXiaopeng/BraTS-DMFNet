# 3D DMFNet for Real-time Brain Tumor Segmentation

This repository is the work of "_3D Dilated Multi-Fiber Network for Real-time Brain Tumor Segmentation in MRI_" based on **pytorch** implementation. You could click the link to access the [paper](https://arxiv.org/pdf/1904.03355.pdf). The multimodal brain tumor dataset (BraTS 2018) could be acquired from [here](https://www.med.upenn.edu/sbia/brats2018.html).

## Dilated multi-fiber network


<div  align="center">  
 <img src="https://github.com/China-LiuXiaopeng/BraTS-DMFNet/blob/master/fig/Architecture.jpg"
     align=center/>
</div>

 <center>Architecture of 3D DMFNet</center>


## Requirements
* python 3.6
* pytorch 0.4 or 1.0
* nibabel
* pickle 

## Implementation

Download the BraTS2018 dataset and change the path:

```
experiments/PATH.yaml
```

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

(Optional) Split the training set into k-fold for the **cross-validation** experiment.

```
python split.py
```

### Training

Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=8 is recommended. The total training time take less than 10 hours in gtxforce 1080Ti.

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

Where ET: the enhancing tumor, WT: the whole tumor, TC: the tumor core.

You could download the trained **DMFnet (pytorch)** from [Google drive](https://drive.google.com/open?id=17C-rbNQZtBoCH1Dgu3wYJQm8N0_DbdxH) or [Baidu dirve](https://pan.baidu.com/s/1dRyo9ZvisZvAwO4TVen2Pg)(password for download: t8xu). Put the weight in the dir './ckpt/DMFNet_GDL_all/here'. You could obtain the resutls as paper reported by running the following code:

```
python test.py --mode=1 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=DMFNet_GDL_all --gpu=0
```
Then make a submission to the online evaluation server.

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
