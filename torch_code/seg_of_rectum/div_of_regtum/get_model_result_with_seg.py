import os
import imageio
import torch
from unet import UNet
from dataset_4layers import Dataset4Layers
# from dataset import Dataset
# from dataset_1layer import Dataset1Layer
# from dataset_3layers import Dataset3Layers
# from emb_dataset import EmbDataset
from torch.utils.data import DataLoader
import numpy as np
from skimage import color
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def display_jpg(img, pred, file_name, train_phrase, display_path):
    pred = np.where(pred > 0.8, 1, 0)
    img = color.gray2rgb(img)
    img[pred == 0, 1] = 0
    img[pred == 0, 2] = 0
    save_path = os.path.join(display_path, train_phrase)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = os.path.join(save_path, file_name)
    imageio.imwrite(file_path, img, 'PNG-FI')

def save_mask(pred, file_name, train_phrase, mask_path):
    pred = np.where(pred > 0.4, 255, 0).astype('uint8')
    save_path = os.path.join(mask_path, train_phrase)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    file_path = os.path.join(save_path, file_name)
    imageio.imwrite(file_path, pred)

def predict(img_path, ori_seg_path, ckpt_path, train_phrase, display_path, mask_path):
    ckpt = os.listdir(ckpt_path)
    ckpt.sort(reverse=True)
    ckpt = ckpt[0]

    model = UNet(channels_in=4, channels_out=1)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt)))

    dataset = Dataset4Layers(img_path, ori_seg_path, img_path, train_phrase)
    dataloder = DataLoader(dataset, batch_size=3, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.float().to(device)
    model.eval()
    
    for step, batch in enumerate(dataloder):
        features = batch['features'].to(device)
        file_name = batch['file_name']
        pred = model(features)
        pred = pred.data.cpu().numpy()
        for i in range(len(file_name)):
            file_path = os.path.join(img_path, train_phrase, file_name[i])
            img = imageio.imread(file_path)
            display_jpg(img, pred[i].squeeze(), file_name[i], train_phrase, display_path)
            save_mask(pred[i].squeeze(), file_name[i], train_phrase, mask_path)

if __name__ == "__main__":
    # out
    display_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/display'
    mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results_with_seg/mask'

    # in
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ori_img'
    ori_seg_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ori_seg'
    # loc_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/location_2Dslice'
    ckpt_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ckpt_with_ori_seg'

    predict(img_path, ori_seg_path, ckpt_path, 'train', display_path, mask_path)
    predict(img_path, ori_seg_path, ckpt_path, 'val', display_path, mask_path)
    predict(img_path, ori_seg_path, ckpt_path, 'test', display_path, mask_path)
    # predict(img_path, ckpt_path, 'test', display_path, mask_path)
    pass
