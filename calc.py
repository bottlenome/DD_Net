import skimage.measure


def normalize(src):
    mi = np.min(src)
    ma = np.max(src)
    return (src - mi) / (ma - mi)


def compare(src, result):
    psnr = skimage.measure.compare_psnr(src, result)
    ssim = skimage.measure.compare_ssim(src, result)


if __name__ == '__main__':
    for i in range(10):

