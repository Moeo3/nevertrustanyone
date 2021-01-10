import os
import numpy as np
import xlwt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data
from dataset import ModelDataSet
import torch
from gugunetv3 import GuguNetV3
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_point(unet, epoch, dice_loss_val, min_loss, pth_path):
    save_freq = 15
    if (epoch % save_freq == 0) or (epoch > 10 and dice_loss_val < min_loss):
        filename = os.path.join(pth_path, 'convnet%d.pth' % (epoch))
        torch.save(unet.state_dict(), filename)
        print('[check_point] model save to %s' % (filename))
        pass
    return

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

def load_model():
    model = GuguNetV3(channels_in=1)
    model.apply(init_weights)
    model = model.cuda()
    loss = nn.SmoothL1Loss()
    return model, loss

def train_gugunet(dataloader, optimizer_G, net, loss):
    net.train()
    loss_l1_total = 0.
    for idx, batch in enumerate(dataloader):

        data = batch['data'].cuda()
        label = batch['label'].cuda()
        #update G
        optimizer_G.zero_grad()
        pred = net(data)
        l1_loss = loss(pred, label)
        loss_l1_total += l1_loss.item()
        l1_loss.backward()
        # ssim_loss.backward()
        optimizer_G.step()
    return loss_l1_total / len(dataloader)

def eval_gugunet(dataloader, net, loss):
    net.eval()
    loss_l1_total = 0.
    for idx, batch in enumerate(dataloader):
        data = batch['data'].cuda()
        label = batch['label'].cuda()
        pred = net(data)
        l1_loss = loss(pred, label)
        loss_l1_total += l1_loss.item()
    return loss_l1_total / len(dataloader)

def main():
    data_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/patch_folder/'
    score_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/score_folder/'
    # label_path = '/data2/local/data/zhangqianru/data/rectal_seg_data/seg_label_2Dslice/'
    model, loss = load_model()
    train_set = ModelDataSet(img_path=data_path, label_path=score_path, train_phase='train', rand_transform=True)
    val_set = ModelDataSet(img_path=data_path, label_path=score_path, train_phase='val', rand_transform=False)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, drop_last=False, num_workers=4)
    optimizer_G = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, milestones=[50, 70], gamma=0.5)
    wb = xlwt.Workbook()
    ws = wb.add_sheet('sheet1')
    min_loss = 1
    pth_path = '/data2/local/data/zhangqianru/data/model_checkpoint_save/rectum_patch_study_2/'
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    for epoch in range(100):
        loss_l1_train = train_gugunet(train_loader, optimizer_G, model, loss)
        loss_l1_val = eval_gugunet(val_loader, model, loss)
        ws.write(epoch, 0, loss_l1_train)
        ws.write(epoch, 1, loss_l1_val)
        print('loss_l1_train: %f, loss_l1_val: %f ' % (loss_l1_train, loss_l1_val))
        scheduler_G.step()
        check_point(model, epoch, loss_l1_val, min_loss, pth_path)
        if min_loss > loss_l1_val:
            min_loss = loss_l1_val
        wb.save('/home/zhangqianru/rectum_patch_study_2.xls')

if __name__ == '__main__':
    main()

