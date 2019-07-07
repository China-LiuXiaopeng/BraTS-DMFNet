import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc

cudnn.benchmark = True

path = os.path.dirname(__file__)


# dice socre is equal to f1 score
def dice_score(o, t,eps = 1e-8):
    num = 2*(o*t).sum() + eps #
    den = o.sum() + t.sum() + eps # eps
    # print(o.sum(),t.sum(),num,den)
    print('All_voxels:240*240*155 | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}'.format(int(num),int(den),o.sum(),int(t.sum())))
    return num/den


def softmax_output_dice(output, target):
    ret = []

    # whole
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core
    o = (output==1) | (output==3)
    t = (target==1) | (target==4)
    ret += dice_score(o , t),
    # active
    o = (output==3); t = (target==4)
    ret += dice_score(o , t),

    return ret

keys = 'whole', 'core', 'enhancing', 'loss'

def validate_softmax(
        valid_loader,
        model,
        cfg='',
        savepath='', # when in validation set, you must specify the path to save the 'nii' segmentation results here
        names=None, # The names of the patients orderly!
        scoring=True, # If true, print the dice score.
        verbose=False,
        use_TTA=False, # Test time augmentation, False as default!
        save_format=None, # ['nii','npy'], use 'nii' as default. Its purpose is for submission.
        snapshot=False, # for visualization. Default false. It is recommended to generate the visualized figures.
        postprocess=False, # Defualt False, when use postprocess, the score of dice_ET would be changed.
        cpu_only=False):

    assert cfg is not None
    H, W, T = 240, 240, 155
    model.eval()
    runtimes = []
    vals = AverageMeter()
    for i, data in enumerate(valid_loader):
        target_cpu = data[1][0, :H, :W, :T].numpy() if scoring else None # when validing, make sure that argument 'scoring' must be false, else it raise a error!

        if cpu_only == False:
            data = [t.cuda(non_blocking=True) for t in data]
        x, target = data[:2]

        # compute output
        if not use_TTA:

            # torch.cuda.synchronize()
            start_time = time.time()
            logit = model(x)
            # torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            runtimes.append(elapsed_time)

            output = F.softmax(logit,dim=1)
        else:
            logit = F.softmax(model(x) ,1) # 000
            logit += F.softmax(model(x.flip(dims=(2,))).flip(dims=(2,)),1)
            logit += F.softmax(model(x.flip(dims=(3,))).flip(dims=(3,) ),1)
            logit += F.softmax(model(x.flip(dims=(4,))).flip(dims=(4,)),1)
            logit += F.softmax(model(x.flip(dims=(2,3))).flip(dims=(2,3) ),1)
            logit += F.softmax(model(x.flip(dims=(2,4))).flip(dims=(2,4)),1)
            logit += F.softmax(model(x.flip(dims=(3,4))).flip(dims=(3,4)),1)
            logit += F.softmax(model(x.flip(dims=(2,3,4))).flip(dims=(2,3,4)),1)
            output = logit / 8.0 # mean

        output = output[0, :, :H, :W, :T].cpu().numpy()
        ############

        output = output.argmax(0) # (channels,height,width,depth)

        if postprocess == True:
            ET_voxels = (output == 3).sum()
            if ET_voxels < 500:
                output[np.where(output == 3)] = 1

        msg = 'Subject {}/{}, '.format(i+1, len(valid_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)

        if savepath:
            # .npy for farthur model ensemble
            # .nii for directly model submission
            assert save_format in ['npy','nii']
            if save_format == 'npy':
                np.save(os.path.join(savepath, name + '_preds'), output)
            if save_format == 'nii':
                oname = os.path.join(savepath,'submission', name + '.nii.gz')
                seg_img = np.zeros(shape=(H,W,T),dtype=np.uint8)

                seg_img[np.where(output==1)] = 1
                seg_img[np.where(output==2)] = 2
                seg_img[np.where(output==3)] = 4
                if verbose:
                    print('1:',np.sum(seg_img==1),' | 2:',np.sum(seg_img==2),' | 4:',np.sum(seg_img==4))
                    print('WT:',np.sum((seg_img==1)|(seg_img==2)|(seg_img==4)),' | TC:',np.sum((seg_img==1)|(seg_img==4)),' | ET:',np.sum(seg_img==4))
                nib.save(nib.Nifti1Image(seg_img, None),oname)

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.uint8)
                    Snapshot_img[:, :, 0, :][np.where(output == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(output == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(output == 3)] = 255

                    for frame in range(T):
                        os.makedirs(os.path.join(savepath,'snapshot',name),exist_ok=True)
                        scipy.misc.imsave(os.path.join(savepath,'snapshot',name,str(frame)+'.png'), Snapshot_img[:,:,:,frame])

        if scoring:
            scores = softmax_output_dice(output, target_cpu)
            vals.update(np.array(scores))
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, scores)])

            if snapshot:
                # red: (255,0,0) green:(0,255,0) blue:(0,0,255) 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.
                gap_width = 2 # boundary width = 2
                Snapshot_img = np.zeros(shape=(H, W*2+gap_width,3,T), dtype=np.uint8)
                Snapshot_img[:,W:W+gap_width,:] = 255 # white boundary

                empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                empty_fig[np.where(output == 1)] = 255
                Snapshot_img[:,:W,0,:] = empty_fig
                empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                empty_fig[np.where(target_cpu == 1)] = 255
                Snapshot_img[:, W+gap_width:, 0, :] = empty_fig

                empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                empty_fig[np.where(output == 2)] = 255
                Snapshot_img[:,:W,1,:] = empty_fig
                empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                empty_fig[np.where(target_cpu == 2)] = 255
                Snapshot_img[:, W+gap_width:, 1, :] = empty_fig

                empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                empty_fig[np.where(output == 3)] = 255
                Snapshot_img[:,:W,2,:] = empty_fig
                empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
                empty_fig[np.where(target_cpu == 4)] = 255
                Snapshot_img[:, W+gap_width:,2, :] = empty_fig

                for frame in range(T):
                    os.makedirs(os.path.join( 'snapshot',cfg, name), exist_ok=True)
                    scipy.misc.imsave(os.path.join('snapshot',cfg, name, str(frame) + '.png'), Snapshot_img[:,:,:,frame])

        logging.info(msg)

    if scoring:
        msg = 'Average scores:'
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(keys, vals.avg)])
        logging.info(msg)

    computational_runtime(runtimes)

    model.train()
    return vals.avg

def computational_runtime(runtimes):
    # remove the maximal value and minimal value
    runtimes = np.array(runtimes)
    maxvalue = np.max(runtimes)
    minvalue = np.min(runtimes)
    nums = runtimes.shape[0] - 2
    meanTime = (np.sum(runtimes) - maxvalue - minvalue ) / nums
    fps = 1 / meanTime
    print('mean runtime:',meanTime,'fps:',fps)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
