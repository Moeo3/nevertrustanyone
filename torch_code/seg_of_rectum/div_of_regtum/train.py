import xlwt, torch, os
from unet import UNet
from dice_loss import DiceLoss
from dataset_1layer import Dataset1Layer
from dataset_3layers import Dataset3Layers
from torch.nn import BatchNorm2d, Conv2d, MSELoss
from torch.nn.init import kaiming_normal_
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from vgg_pretrained import VggPretrained

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def init_weight(model):
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
        elif isinstance(m, Conv2d):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

def load_model_and_dataset(model_name):
    if model_name.find('3layers') >= 0:
        model = UNet(channels_in=3, channels_out=1)
        Dataset = Dataset3Layers
    else:
        model = UNet(channels_in=1, channels_out=1)
        Dataset = Dataset1Layer
    model.apply(init_weight)
    return model, Dataset

def epoch_step(dataloder, net, opt, vgg_tag, loss, train_phrase, device):
    if train_phrase == 'train':
        train_tag = True
        net.train()
    else:
        train_tag = False
        net.eval()

    if vgg_tag and train_tag:
        vgg = VggPretrained().to(device)
        mse_loss = MSELoss()

    dice_loss_total = 0.

    for step, batch in enumerate(dataloder):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        if train_tag:
            opt.zero_grad()
        pred = net(features)
        if vgg_tag and train_tag:
            label1, label2, label3, label4, label5 = vgg(labels)
            pred1, pred2, pred3, pred4, pred5 = vgg(pred)
            feature_loss = 0.1 * (mse_loss(pred1, label1) + mse_loss(pred2, label2) + mse_loss(pred3, label3)
                                + mse_loss(pred4, label4) + mse_loss(pred5, label5))
            feature_loss.backward(retain_graph=True)
        dice_loss = loss(labels, pred)
        dice_loss_total = dice_loss_total + dice_loss.item()
        if train_tag:
            dice_loss.backward()
            opt.step()
    return dice_loss_total / len(dataloder)

def save_ckpt(ckpt_path, model_name, epoch, dict):
    ckpt_path = os.path.join(ckpt_path, model_name)
    epoch = str(epoch).zfill(2)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    save_path = os.path.join(ckpt_path, f'epoch{epoch}.pth')
    torch.save(dict, save_path)

def train(img_path, label_path, ckpt_path, xls_path, ws, model, model_name, Dataset, vgg_tag):
    train_set = Dataset(img_path, label_path, 'train')
    train_loader = DataLoader(train_set, batch_size=3, shuffle=True)
    val_set = Dataset(img_path, label_path, 'val')
    val_loader = DataLoader(val_set, batch_size=3, shuffle=False)

    opt = Adam(model.parameters(), 1e-4, betas=(0.9, 0.999))
    sch = StepLR(opt, step_size=30, gamma=0.8)
    loss = DiceLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.float().to(device)

    max_epoch = 151
    # max_epoch = 2
    cnt = 0
    stop_count = 15
    min_dice_loss = 1.
    stop_flag = False
    for i in range(max_epoch):
        dice_loss_train = epoch_step(train_loader, model, opt, vgg_tag, loss, 'train', device)
        # dice_loss_train = epoch_step(val_loader, model, opt, vgg_tag, loss, 'train', device)
        dice_loss_val = epoch_step(val_loader, model, opt, vgg_tag, loss, 'val', device)
        loss_list = [dice_loss_train, dice_loss_val]
        for j in range(len(loss_list)):
            ws.write(i, j, loss_list[j])

        print(f'in epoch{i}: train dice loss is {dice_loss_train}, val dice loss is {dice_loss_val}')

        if dice_loss_val < min_dice_loss:
            min_dice_loss = dice_loss_val
            save_ckpt(ckpt_path, model_name, i, model.state_dict())
            cnt = 0
        else:
            cnt = cnt + 1
        if cnt == stop_count:
            stop_flag = True
            break
        sch.step()

    if not stop_flag:
        save_ckpt(ckpt_path, model_name, max_epoch - 1, model.state_dict())
    return ws

def train_nets(img_path, label_path, ckpt_path, xls_path, model_set):
    for i in range(len(model_set)):
        wb = xlwt.Workbook()
        ws = wb.add_sheet('diceloss')
        model_name = model_set[i]
        model, Dataset = load_model_and_dataset(model_name)
        if model_name.find('vgg_loss') >= 0:
            vgg_tag = True
        else:
            vgg_tag = False
        ws = train(img_path, label_path, ckpt_path, xls_path, ws, model, model_name, Dataset, vgg_tag)
        wb.save(os.path.join(xls_path, f'seg_of_rectum_{model_name}.xls'))

if __name__ == "__main__":
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/origin_img_2Dslice'
    label_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/seg_label_2Dslice'
    ckpt_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ckpt'
    xls_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/xls'
    # model_set = ['unet', 'unet_3layers', 'unet_3layers_with_vgg_loss', 'unet_with_vgg_loss']
    # model_set = ['unet', 'unet_3layers']
    # model_set = ['unet']
    # model_set = ['unet_with_vgg_loss']
    model_set = ['unet_3layers']

    train_nets(img_path, label_path, ckpt_path, xls_path, model_set)
