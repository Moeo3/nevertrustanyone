import os, imageio
import numpy as np
from skimage import measure, morphology
import math

pi = 3.14159265358979

def select_label(img, file_name):
    labels = measure.label(img, connectivity=2)
    if labels.max() == 1:
        img = img.astype('bool')
        img = morphology.convex_hull_image(img)
        img = morphology.binary_dilation(img, morphology.disk(3))
        return img
    props = measure.regionprops(labels)
    areas = list(map(lambda x: x.area, props))
    max_area = np.max(areas)
    min_area = int(max_area / 1.3)
    img = morphology.remove_small_objects(labels, min_size=min_area, connectivity=1)
    img = img.astype('bool')
    labels = measure.label(img, connectivity=2)
    if labels.max() == 1:
        img = img.astype('bool')
        img = morphology.convex_hull_image(img)
        props = measure.regionprops(labels)
        areas = list(map(lambda x: x.area, props))
        r = math.sqrt(areas[0] / pi)
        img = morphology.binary_dilation(img, morphology.disk(int(r / 3)))
    else:
        img = img.astype('bool')
        with open('/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/special_val', 'a') as f:
            f.write(file_name + '\n')
    return img

def simple_post(mask_path, post_path, train_phrase):
    path = os.path.join(mask_path, train_phrase)
    save_path = os.path.join(post_path, train_phrase)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        img = imageio.imread(file_path)
        if img.max() == 0:
            continue
        img = img.astype('bool')
        img = select_label(img, file_name).astype('uint8')
        img = img * 255
        imageio.imwrite(os.path.join(save_path, file_name), img)

if __name__ == "__main__":
    mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/mask'
    post_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/post_mask'
    if not os.path.exists(post_path):
        os.mkdir(post_path)
    simple_post(mask_path, post_path, 'val')
    pass
