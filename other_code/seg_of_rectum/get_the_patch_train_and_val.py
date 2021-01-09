import os
import numpy as np
import imageio

def get_the_patch(num, img, mask, cancer_seg, save_path, score_path, patch_size=48, step_width=8):
    maxh, maxw = img.shape[0], img.shape[1]
    site = np.where(mask > 0)
    high, low, left, right = site[0].min(), site[0].max(), site[1].min(), site[1].max()
    for i in range(high, low, step_width):
        for j in range(left, right, step_width):
            a, b, c, d = i, i + patch_size, j, j + patch_size
            if b > min(maxh, low + step_width) or d > min(maxw, right + step_width):
                continue
            patch_mask = np.zeros(img.shape)
            patch_mask[a:b, c:d] = 1
            and_set = patch_mask + mask
            if and_set.max() == mask.max():
                continue
            score = sum(sum(cancer_seg[a:b, c:d] / 255)) / (patch_size * patch_size)
            patch = img[a:b, c:d]
            imageio.imwrite(os.path.join(save_path, f'{num}.png'), patch)
            with open(score_path, 'a') as f:
                f.write(f'{num}, {score} \n')
            num = num + 1
    return num

def find_patchs(img_path, seg_path, mask_path, patch_path, score_path, train_phrase):
    img_path = os.path.join(img_path, train_phrase)
    seg_path = os.path.join(seg_path, train_phrase)
    mask_path = os.path.join(mask_path, train_phrase)
    patch_path = os.path.join(patch_path, train_phrase)
    if not os.path.exists(patch_path):
        os.mkdir(patch_path)
    score_path = os.path.join(score_path, f'{train_phrase}.csv')

    files = os.listdir(mask_path)
    files.sort()
    num = 0
    for file_name in files:
        seg = imageio.imread(os.path.join(seg_path, file_name))
        if seg.max() == 0:
            continue
        img = imageio.imread(os.path.join(img_path, file_name))
        mask = imageio.imread(os.path.join(mask_path, file_name))
        num = get_the_patch(num, img, mask, seg, patch_path, score_path)
        

if __name__ == "__main__":
    # in
    cancer_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/cancer_seg'
    # fibro_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/fibro_seg'
    div_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/div_result'
    img_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/ori_img'

    # out
    patch_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/patch_folder'
    score_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/score_folder'

    find_patchs(img_path, cancer_path, div_path, patch_path, score_path, 'train')
    find_patchs(img_path, cancer_path, div_path, patch_path, score_path, 'val')
    pass
