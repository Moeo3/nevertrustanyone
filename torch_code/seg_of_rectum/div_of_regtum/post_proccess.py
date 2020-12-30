import os
import imageio
from itertools import groupby
from skimage import measure, morphology
import numpy as np

def dilation(img):
    labels = measure.label(img, connectivity=2)
    # 无联通块
    if labels.max() != 1:
        return 0, False
    props = measure.regionprops(labels)
    areas = list(map(lambda x: x.area, props))
    area = np.max(areas)

    if area < 6077:
        pixel = 3
    elif area > 8181:
        pixel = 15
    else:
        pixel = 9
    
    return morphology.binary_dilation(img, morphology.disk(pixel)), True

def select_label(img):
    labels = measure.label(img, connectivity=2, background=0)
    if labels.max() == 0:
        return img, True
    if labels.max() == 1:
        return img, True
    props = measure.regionprops(labels)
    areas = list(map(lambda x: x.area, props))
    max_area = np.max(areas)
    min_area = int(max_area / 1.2)
    img = morphology.remove_small_objects(labels, min_size=min_area, connectivity=1)
    labels = measure.label(img, connectivity=2, background=0)
    props = measure.regionprops(labels)
    if labels.max() == 1:
        return img, True
    else:
        return img, False

def calc_and_save(select_list, save_file_path):
    and_set = np.zeros(select_list[0].shape).astype('bool')
    for select in select_list:
        and_set = and_set + select
    and_set = np.where(and_set > 0, 1, 0)
    and_set = morphology.convex_hull_image(and_set)
    img, flag = dilation(and_set)
    # 无联通块
    if not flag:
        return False
    imageio.imwrite(save_file_path, (img * 255).astype('uint8'))
    # 保存成功
    return True

def proccessing(model_res_path, save_path, model_name, train_phrase):
    mask_path = os.path.join(model_res_path, train_phrase, model_name)
    png_list = os.listdir(mask_path)
    png_list.sort()
    for i in range(len(png_list)):
        file_name = png_list[i]
        save_name = file_name
        save_file_path = os.path.join(save_path, train_phrase, model_name, save_name)
        file_name_split = [''.join(list(g)) for k, g in groupby(file_name, key=lambda x: x.isdigit())]
        patient_name = file_name_split[0]
        # if patient_name == 'zhaomeizhu':
        #     a = 1
        #     print('?????')
        if i == 0:
            last_file_name = file_name
            file_name = png_list[i + 1]
            next_file_name = png_list[i + 2]
        elif i == (len(png_list) - 1):
            last_file_name = png_list[i - 2]
            file_name = png_list[i - 1]
            next_file_name = png_list[i]
        else:
            last_file_name = png_list[i - 1]
            next_file_name = png_list[i + 1]
        last_file_name_split = [''.join(list(g)) for k, g in groupby(last_file_name, key=lambda x: x.isdigit())]
        next_file_name_split = [''.join(list(g)) for k, g in groupby(next_file_name, key=lambda x: x.isdigit())]
        if last_file_name_split[0] != patient_name:
            last_file_name = file_name
            file_name = png_list[i + 1]
            next_file_name = png_list[i + 2]
        if next_file_name_split[0] != patient_name:
            next_file_name = file_name
            file_name = png_list[i - 1]
            last_file_name = png_list[i - 2]
        
        last_img = (imageio.imread(os.path.join(mask_path, last_file_name)) / 255).astype('bool')
        img = (imageio.imread(os.path.join(mask_path, file_name)) / 255).astype('bool')
        next_img = (imageio.imread(os.path.join(mask_path, next_file_name)) / 255).astype('bool')

        img_list = [last_img, img, next_img]
        select_list = []
        multi_flag = False
        for img in img_list:
            select, flag = select_label(img)
            if flag is False:
                multi_flag = True
            select_list.append(select)

        # 只有一个联通块或没有联通块
        if multi_flag is False:
            flag = calc_and_save(select_list, save_file_path)
            # 无联通块
            if not flag:
                pass
            continue
        # 有多个联通块
        else:
            inter_set = np.zeros(select_list[0].shape).astype('bool')
            for select in select_list:
                inter_set = inter_set + select
            inter_set = np.where(inter_set == len(select_list), 1, 0)
            select_inter_set, flag = select_label(inter_set)
            # 无联通块或并集有多个联通块
            if inter_set.max() == 0 or flag is False:
                pass
            else:
                # 并集只有一个联通块
                label_img = measure.label(select_inter_set)
                props = measure.regionprops(label_img)
                centroid = props[0].centroid
                x = int(centroid[0])
                y = int(centroid[1])
                for select in select_list:
                    color = select[x][y]
                    select = np.where(select == color, 1, 0)
                flag = calc_and_save(select_list, save_file_path)
            continue
        # 不处理了……
        ori_img = (imageio.imread(os.path.join(mask_path, save_name)) / 255).astype('bool')
        img, flag = select_label(ori_img)
        img = morphology.convex_hull_image(img)
        img, flag = dilation(img)
        imageio.imwrite(save_file_path, (img * 255).astype('uint8'))

if __name__ == "__main__":
    model_res_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results/mask'
    save_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/mask_post'
    # model_set = ['unet', 'unet_3layers', 'unet_3layers_with_vgg_loss', 'unet_with_vgg_loss', 'emb']
    model_set = ['unet']

    for model in model_set:
        # proccessing(model_res_path, save_path, model, 'train')
        proccessing(model_res_path, save_path, model, 'val')

    pass
