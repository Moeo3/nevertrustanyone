import os, imageio
import numpy as np
from skimage import measure, morphology
import math

pi = 3.14159265358979
file_names = ['liuchanglan04.png', 'yinshaozeng12.png', 'linzhiqiang10.png', 'guoqingyuan15.png', 'linzhiqiang11.png', 'zhengzequan12.png', 'linliping02.png', 'linzhiqiang12.png', 'chenyingqi01.png', 'quxiaomei15.png', 'lanfangsheng15.png', 'dengfuqing19.png', 'sunyunwu20.png', 'lanfangsheng16.png', 'sunyunwu19.png', 'songzhi08.png', 'huangxiaoqiang06.png', 'sunyunwu18.png', 'liuguigen12.png', 'sunyunwu16.png', 'zhangshengtian13.png', 'wugenghui14.png', 'zhangshengtian12.png', 'sunyunwu17.png', 'weixiuming14.png', 'yuanzhongwen14.png', 'weixiuming15.png', 'zhangwenhong02.png', 'zhangwenhong03.png', 'wugenghui13.png', 'zhangshengtian14.png', 'dongjinfang06.png', 'zhangjianbai16.png', 'zhangwenhong01.png', 'zhangjianbai17.png', 'wushanzhen14.png', 'linchuxiang16.png', 'wushanzhen12.png', 'wushanzhen06.png', 'yuanying05.png', 'fuwenqing20.png', 'zhangguihua12.png', 'wushanzhen07.png', 'wushanzhen13.png', 'wushanzhen05.png', 'zhangguihua11.png']

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
        with open('/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/special_1', 'a') as f:
            f.write(file_name + '\n')
    return img

def simple_post(mask_path, post_path, train_phrase):
    path = os.path.join(mask_path, train_phrase)
    save_path = os.path.join(post_path, train_phrase)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        img = imageio.imread(file_path)
        if img.max() == 0:
            continue
        img = img.astype('bool')
        img = select_label(img, file_name).astype('uint8')
        img = img * 255
        imageio.imwrite(os.path.join(save_path, file_name), img)

if __name__ == "__main__":
    mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/post_mask'
    post_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/post_mask'
    if not os.path.exists(post_path):
        os.mkdir(post_path)
    simple_post(mask_path, post_path, 'test')
    pass
