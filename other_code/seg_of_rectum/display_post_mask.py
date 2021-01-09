import os, imageio
from skimage import color

def display_post(img_path, post_mask_path, post_display_path, train_phrase):
    img_path = os.path.join(img_path, train_phrase)
    post_display_path = os.path.join(post_display_path, train_phrase)
    if not os.path.exists(post_display_path):
        os.mkdir(post_display_path)
    post_mask_path = os.path.join(post_mask_path, train_phrase)

    for file_name in os.listdir(post_mask_path):
        if not file_name.endswith('.png'):
            continue
        img = imageio.imread(os.path.join(img_path, file_name))
        img = color.gray2rgb(img)
        mask = imageio.imread(os.path.join(post_mask_path, file_name))
        img[mask == 0, 1] = 0
        img[mask == 0, 2] = 0
        imageio.imwrite(os.path.join(post_display_path, file_name), img)

if __name__ == "__main__":
    post_mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/post_mask'
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ori_img'
    post_display_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/post_display'
    if not os.path.exists(post_display_path):
        os.mkdir(post_display_path)
    display_post(img_path, post_mask_path, post_display_path, 'train')
    display_post(img_path, post_mask_path, post_display_path, 'val')
    display_post(img_path, post_mask_path, post_display_path, 'test')
    pass
