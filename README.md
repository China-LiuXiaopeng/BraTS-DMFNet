# 3D DMFNet for Real-time BraTS 2018 segmentation

This repository is the work of "3D Dilated Multi-Fiber Network for Real-time Brain Tumor Segmentation in MRI". You could click the Link to access the [paper]("https://arxiv.org/pdf/1904.03355.pdf"). The Multimodal Brain Tumor Dataset (2018) could be acquired from [here]("https://www.med.upenn.edu/sbia/brats2018.html").


## Dilated multi-fiber Network

![Image text](https://github.com/China-LiuXiaopeng/BraTS-DMFNet/blob/master/fig/Architecture.jpg)

## Requirements
* python 3.6
* pytorch 0.4 or 1.0
* nibabel
* pickle 

## Implementation

Download the BraTS2018 dataset and change the path in ./experiments/PATH.yaml.

###Data preprocess
```
python preprocess.py
```
###Training
```
python train_all.py --gpu=0,1,2,3 --cfg=DMFNet_GDL_all --batch_size=8
```

###Test
We have provided the trained weights for download (Baidu drive). You could obtain the results as paper reported. Please save the weights in the ./ckpt/dir/here. We will upload the weights to the  google drive soon.

| Model         | Params (M) | Dice_ET (%) | Dice_WT (%) | Dice_TC (%) | PyTorch Model |
| :-------------|:----------------:|:----------------------------------: |
| 0.75x MFNet   | 1.81 | 79.34 | 90.22 | 84.25 | [here]("https://pan.baidu.com/s/1X5FWuG3Z93hBvXp8Pje73Q") (提取码: zmkm) |
| MFNet         | 3.19 | 79.91 | 90.43 | 84.61 | [here]("https://pan.baidu.com/s/1if2rfnjKCgWHvBvumvGWJA") (提取码: j66m) |
| DMFNet        | 3.88 | 80.12 | 90.62 | 84.54 | [here]("https://pan.baidu.com/s/1dRyo9ZvisZvAwO4TVen2Pg") (提取码: t8xu) |

Then run the testing code:
```
python test.py --mode=0 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=DMFNet_GDL_all --gpu=0
```
## Citation
If you use our code/model in your work or find it is helpful, please cite the paper:
```
@inproceedings{chen2019dmfnet,
  title={3D Dilated Multi-Fiber Network for Real-time Brain Tumor Segmentation in MRI},
  author={Chen, Chen and Liu, Xiaopeng and Ding, Meng and Zheng, Junfeng and Li, Jiangyun},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2019}
}
```
##Thanks
1. [MFNet]("https://github.com/cypw/PyTorch-MFNet")
2. [BraTS2018-tumor-segmentation]("https://github.com/ieee820/BraTS2018-tumor-segmentation")
