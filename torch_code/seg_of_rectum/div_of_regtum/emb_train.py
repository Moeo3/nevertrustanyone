from dice_loss import DiceLoss
from unet import UNet
from emb_dataset import EmbDataset
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
from torch.utils.data import DataLoader
import xlwt
from torch.nn import BatchNorm2d, Conv2d
from torch.nn.init import kaiming_normal_

def init_weight(model):
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
        elif isinstance(m, Conv2d):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

def save_ckpt(ckpt_path, model_name, epoch, dict):
    ckpt_path = os.path.join(ckpt_path, model_name)
    epoch = str(epoch).zfill(2)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    save_path = os.path.join(ckpt_path, f'epoch{epoch}.pth')
    torch.save(dict, save_path)

def epoch_step(net, dataloader, opt, loss, train_phrase, device):
    if train_phrase == 'train':
        net.train()
        train_tag = True
    else:
        net.eval()
        train_tag = False
    
    dice_loss_total = 0.
    for step, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        if train_tag:
            opt.zero_grad()
        pred = net(features)
        dice_loss = loss(labels, pred)
        dice_loss_total = dice_loss_total + dice_loss.item()
        if train_tag:
            dice_loss.backward()
            opt.step()
    return dice_loss_total / len(dataloader)

def re_train(net, train_dataloader, val_dataloader, ckpt_path, xls_path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('dice_loss')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.float().to(device)
    net.apply(init_weight)
    opt = Adam(net.parameters(), lr=2e-4)
    sch = StepLR(opt, step_size=10, gamma=0.5)
    loss = DiceLoss()

    max_epoch = 101
    cnt = 0
    stop_cnt = 10
    min_dice_loss = 1.
    stop_flag = False
    for i in range(max_epoch):
        train_dice_loss = epoch_step(net, train_dataloader, opt, loss, 'train', device)
        val_dice_loss = epoch_step(net, val_dataloader, opt, loss, 'val', device)
        loss_list = [train_dice_loss, val_dice_loss]
        print(f'in epoch{i}: train dice loss is {train_dice_loss}, test dice loss is {val_dice_loss}')
        for j in range(len(loss_list)):
            ws.write(i, j, loss_list[j])
        if val_dice_loss < min_dice_loss:
            min_dice_loss = val_dice_loss
            save_ckpt(ckpt_path, 'emb', i, net.state_dict())
            cnt = 0
        else:
            cnt = cnt + 1
        if cnt == stop_cnt:
            stop_flag = True
            break
        sch.step()
    if not stop_flag:
        save_ckpt(ckpt_path, 'emb', i, net.state_dict())
    wb.save(os.path.join(xls_path, 'seg_of_rectum_emb.xls'))

if __name__ == "__main__":
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/origin_img_2Dslice'
    mask_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/seg_label_2Dslice'
    model_res_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/model_results/mask'
    ckpt_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ckpt'
    xls_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/xls'
    model_set = ['unet', 'unet_3layers', 'unet_3layers_with_vgg_loss', 'unet_with_vgg_loss']

    train_dataset = EmbDataset(img_path, mask_path, model_res_path, model_set, 'train')
    val_dataset = EmbDataset(img_path, mask_path, model_res_path, model_set, 'val')
    channels_in = len(train_dataset.model_set) + 1
    net = UNet(channels_in, 1)
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    re_train(net, train_dataloader, val_dataloader, ckpt_path, xls_path)

    pass
