# 3D DMFNet for Real-time BraTS 2018 segmentation

This repository is the work of "3D Dilated Multi-Fiber Network for Real-time Brain Tumor Segmentation in MRI". Link to the [paper]("https://arxiv.org/pdf/1904.03355.pdf")

## Requirements
* python 3.6
* pytorch 0.4 or 1.0
* nibabel
* pickle 


## Implementation

###test
python test.py --mode=0 --is_out=False --verbose=True --use_TTA=False --postprocess=False --snapshot=False --restore=model_last.pth --cfg=DMFNet_GDL_all --gpu=0

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
