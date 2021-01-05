import os
import imageio
from skimage import color

def display(img_path, mask_path, save_path, model_name, train_phrease):
    img_file_path = os.path.join(img_path, train_phrease)
    mask_file_path = os.path.join(mask_path, train_phrease, model_name)
    save_file_path = os.path.join(save_path, train_phrease, model_name)
    for file_name in os.listdir(img_file_path):
        img = imageio.imread(os.path.join(img_file_path, file_name))
        mask = imageio.imread(os.path.join(mask_file_path, file_name))
        img = color.grey2rgb(img)
        img[mask == 255, 0] = 0
        img[mask == 255, 2] = 0
        imageio.imwrite(os.path.join(save_file_path, file_name), img)
    pass

if __name__ == "__main__":
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/origin_img_2Dslice'
    mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/mask_post'
    save_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/post_display'
    model_set = ['unet_3layers']
    for model in model_set:
        display(img_path, mask_path, save_path, model, 'train')
        display(img_path, mask_path, save_path, model, 'val')
    pass
