import numpy as np
from skimage.metrics import structural_similarity


def MAE(pred, true):
    return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    

def MSE(pred, true):
    return np.mean((pred - true) ** 2, axis=(0, 1)).sum()


def PSNR(pred, true, min_max_norm=True):
    """Peak Signal-to-Noise Ratio.
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:
            return 20. * np.log10(1. / np.sqrt(mse))
        else:
            return 20. * np.log10(255. / np.sqrt(mse))
        

def cal_psnr(pred, true):
    psnr = 0
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            psnr += PSNR(pred[b, f], true[b, f])
    return psnr / (pred.shape[0] * pred.shape[1])


def cal_ssim(pred, true):
    ssim = 0
    for b in range(pred.shape[0]):
        for f in range(pred.shape[1]):
            ssim += structural_similarity(
                pred[b, f].swapaxes(0, 2),
                true[b, f].swapaxes(0, 2), multichannel=True
            )
    return ssim / (pred.shape[0] * pred.shape[1])