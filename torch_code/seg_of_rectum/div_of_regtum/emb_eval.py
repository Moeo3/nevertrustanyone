import os
from network import DiceLoss, UNet
import imageio
import numpy as np
from skimage import color
import SimpleITK as sitk
from torch.utils.data import DataLoader
import torch
from emb_dataset import EmbDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def display(name, mask, img, save_path = '/home/zhangqianru/data/rectal_seg_data/emb_mask/train'):
    try:
        os.mkdir(save_path)
    except:
        pass
    img = (img / img.max() * 255).astype('uint8')
    rgb_img = color.gray2rgb(img)
    # rgb_img[mask == 1, 0] = 0
    # rgb_img[mask == 1, 1] = 255
    # rgb_img[mask == 1, 2] = 0

    imageio.imwrite(os.path.join(save_path, f'{name}.jpg'), rgb_img)
    pass

def eval_net(net, dataloader, train_phrase, save_path = '/home/zhangqianru/emb', img_path = '/home/zhangqianru/data/rectal_seg_data/origin_img_2Dslice/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.float().to(device)
    net.eval()
    loss = DiceLoss()
    aver_dice = 0
    for step, batch in enumerate(dataloader):
        feature = batch['feature'].to(device)
        label = batch['label'].to(device)
        pred = net(feature)
        dice_loss = loss(label, pred)
        aver_dice = aver_dice + dice_loss.item()
        print(f'Dice loss in step {step} is {dice_loss}')
        for i in range(len(label)):
            mask = pred[i, 0, :, :].data.cpu().numpy()
            mask = np.where(mask > 0.4, 1, 0)
            name = batch['name'][i]
            img = sitk.ReadImage(os.path.join(img_path, train_phrase, name))
            img = sitk.GetArrayFromImage(img)
            display(name.split('.')[0], mask, mask)
            # display(name.split('.')[0], mask, img)
            pass
        pass
    aver_dice = aver_dice / len(dataloader)
    print(f'average dice is {aver_dice}.')
    pass

if __name__ == "__main__":
    dataset = EmbDataset(train_phrase='train')
    channels_in = len(dataset.model_set) + 1
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    state = torch.load('/home/zhangqianru/data/ly/ckpt_folder/retrain_2/epoch4.pth')
    epoch = state['epoch']
    print(f'Load epoch {epoch}.')
    net = UNet(channels_in, 1)
    net.load_state_dict(state['net'])
    eval_net(net, dataloader, dataset.train_phrase, save_path='/home/zhangqianru/')
    pass