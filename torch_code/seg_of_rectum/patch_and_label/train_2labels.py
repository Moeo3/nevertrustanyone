import os
import numpy as np
import xlwt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data
from dataset_2labels import Dataset2Labels
import torch
from gugunetv3_2labels import GuguNetV3
from torch.nn import BatchNorm2d, Conv2d, init, BCEWithLogitsLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_point(unet, epoch, dice_loss_val, min_loss, pth_path):
    save_freq = 15
    if (epoch % save_freq == 0) or (epoch > 10 and dice_loss_val < min_loss):
        filename = os.path.join(pth_path, 'convnet%d.pth' % (epoch))
        torch.save(unet.state_dict(), filename)
        print('[check_point] model save to %s' % (filename))
    return

def init_weights(model):
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
        if isinstance(m, Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

def load_model():
    model = GuguNetV3(channels_in=1)
    model.apply(init_weights)
    model = model.cuda()
    loss = BCEWithLogitsLoss()
    return model, loss

def train_gugunet(dataloader, optimizer_G, net, loss):
    net.train()
    loss_total = 0.
    for idx, batch in enumerate(dataloader):
        data = batch['features'].cuda()
        label = batch['labels'].cuda()
        optimizer_G.zero_grad()
        pred = net(data)
        bce_loss = loss(pred, label)
        print(f'[train] in step {idx}, bce loss is {bce_loss.item()}')
        loss_total += bce_loss.item()
        bce_loss.backward()
        optimizer_G.step()
    return loss_total / len(dataloader)

def eval_gugunet(dataloader, net, loss):
    net.eval()
    loss_total = 0.
    for idx, batch in enumerate(dataloader):
        data = batch['data'].cuda()
        label = batch['label'].cuda()
        pred = net(data)
        bce_loss = loss(pred, label)
        loss_total += bce_loss.item()
    return loss_total / len(dataloader)

def main():
    data_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/patch_folder/'
    score_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/score_folder/'
    pth_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/rectum_patch_study_ckpt_2/'
    xls_path = '/home/zhangqianru/data/seg_of_rectum/patch_and_label/rectum_patch_study_2.xls'

    model, loss = load_model()
    train_set = Dataset2Labels(data_path, score_path, 'train')
    val_set = Dataset2Labels(data_path, score_path, 'val')
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, drop_last=False, num_workers=4)
    optimizer_G = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, milestones=[50, 70], gamma=0.5)

    wb = xlwt.Workbook()
    ws = wb.add_sheet('sheet1')
    min_loss = 1.
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    for epoch in range(100):
        loss_train = train_gugunet(train_loader, optimizer_G, model, loss)
        loss_val = eval_gugunet(val_loader, model, loss)
        ws.write(epoch, 0, loss_train)
        ws.write(epoch, 1, loss_val)
        print('loss_train: %f, loss_val: %f ' % (loss_train, loss_val))
        scheduler_G.step()
        check_point(model, epoch, loss_val, min_loss, pth_path)
        if min_loss > loss_val:
            min_loss = loss_val
        wb.save(xls_path)

if __name__ == '__main__':
    main()
