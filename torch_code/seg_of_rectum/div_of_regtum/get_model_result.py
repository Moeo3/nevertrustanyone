import os
import imageio
import torch
from unet import UNet
from dataset_2layers import Dataset2Layers
from dataset_4layers import Dataset4Layers
from emb_dataset import EmbDataset
from torch.utils.data import DataLoader
import numpy as np
from skimage import color

def display_jpg(img, pred, file_name, model_name, train_phrase, display_path):
    pred = np.where(pred > 0.4, 1, 0)
    img = color.gray2rgb(img)
    img[pred == 1, 0] = 0
    img[pred == 1, 1] = img.max()
    img[pred == 1, 2] = 0
    save_path = os.path.join(display_path, train_phrase, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = os.path.join(save_path, file_name)
    imageio.imwrite(file_path, img, 'PNG-FI')

def save_mask(pred, file_name, model_name, train_phrase, mask_path):
    pred = np.where(pred > 0.4, 255, 0).astype('uint8')
    save_path = os.path.join(mask_path, train_phrase, model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = os.path.join(save_path, file_name)
    imageio.imwrite(file_path, pred)

def predict(img_path, ckpt_path, model_name, train_phrase, display_path, mask_path, model_res_path):
    ckpt_path_ = os.path.join(ckpt_path, model_name)
    ckpt = os.listdir(ckpt_path_)
    ckpt.sort(reverse=True)
    ckpt = ckpt[0]

    if model_name.find('3layers') >= 0:
        model = UNet(channels_in=4, channels_out=1)
        Dataset = Dataset4Layers
    elif model_name == 'emb':
        model = UNet(channels_in=5, channels_out=1)
        Dataset = EmbDataset
    else:
        model = UNet(channels_in=2, channels_out=1)
        Dataset = Dataset2Layers

    model.load_state_dict(torch.load(os.path.join(ckpt_path, model_name, ckpt)))

    if model_name == 'emb':
        set = EmbDataset(img_path, img_path, model_res_path, ['unet', 'unet_3layers', 'unet_3layers_with_vgg_loss', 'unet_with_vgg_loss'], train_phrase)
    else:
        set = Dataset(img_path, ori_seg_path, img_path, train_phrase)
    loader = DataLoader(set, batch_size=3, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.float().to(device)
    model.eval()
    
    for step, batch in enumerate(loader):
        features = batch['features'].to(device)
        file_name = batch['file_name']
        pred = model(features)
        pred = pred.data.cpu().numpy()
        for i in range(len(file_name)):
            file_path = os.path.join(img_path, train_phrase, file_name[i])
            img = imageio.imread(file_path)
            display_jpg(img, pred[i].squeeze(), file_name[i], model_name, train_phrase, display_path)
            save_mask(pred[i].squeeze(), file_name[i], model_name, train_phrase, mask_path)

if __name__ == "__main__":
    display_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results/display'
    mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results/mask'
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/origin_img_2Dslice'
    ori_seg_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/original_seg_2Dslice'
    model_res_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results/mask'
    ckpt_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ckpt'
    # model_set = ['unet', 'unet_3layers', 'unet_3layers_with_vgg_loss', 'unet_with_vgg_loss']
    # model_set = ['unet_3layers', 'unet_3layers_with_vgg_loss']
    model_set = ['emb']

    for model in model_set:
        predict(img_path, ckpt_path, model, 'train', display_path, mask_path, model_res_path)
        predict(img_path, ckpt_path, model, 'val', display_path, mask_path, model_res_path)
    pass
