import os
import imageio
from skimage import morphology
import numpy as np

def get_convex(ori_seg_path, save_path):
    for file_name in os.listdir(ori_seg_path):
        img = imageio.imread(os.path.join(ori_seg_path, file_name))
        img = np.where(img > 0, True, False)
        convex = morphology.convex_hull_image(img).astype('uint8')
        convex = np.where(convex > 0, 255, 0).astype('uint8')
        imageio.imwrite(os.path.join(save_path, file_name), convex)


if __name__ == "__main__":
    ori_seg_path = '/home/zhangqianru/data/seg_of_rectum/ori_seg_slice'
    save_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/original_seg_2Dslice/all'
    get_convex(ori_seg_path, save_path)
    pass
