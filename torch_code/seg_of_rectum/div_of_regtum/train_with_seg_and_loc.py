import xlwt, torch, os
from unet import UNet
from dice_loss import DiceLoss
from dataset import Dataset
from torch.nn import BatchNorm2d, Conv2d, MSELoss
from torch.nn.init import kaiming_normal_
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def init_weight(model):
    for m in model.modules():
        if isinstance(m, BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
        elif isinstance(m, Conv2d):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

def epoch_step(dataloder, net, opt, loss, train_phrase, device):
    if train_phrase == 'train':
        train_tag = True
        net.train()
    else:
        train_tag = False
        net.eval()

    dice_loss_total = 0.
    for step, batch in enumerate(dataloder):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        if train_tag:
            opt.zero_grad()
        pred = net(features)
        dice_loss = loss(labels, pred)
        dice_loss_total = dice_loss_total + dice_loss.item()
        # print(f'in step {step}: dice loss is {dice_loss.item()}')
        if train_tag:
            dice_loss.backward()
            opt.step()
    return dice_loss_total / len(dataloder)

def save_ckpt(ckpt_path, epoch, dict):
    epoch = str(epoch).zfill(2)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    save_path = os.path.join(ckpt_path, f'epoch{epoch}.pth')
    torch.save(dict, save_path)

def train(img_path, ori_seg_path, location_path, label_path, ckpt_path, xls_path):
    wb = xlwt.Workbook()
    ws = wb.add_sheet('dice loss')

    model = UNet(channels_in=2, channels_out=1)
    model.apply(init_weight)

    train_set = Dataset(img_path, ori_seg_path, location_path, label_path, 'train')
    train_loader = DataLoader(train_set, batch_size=3, shuffle=True)
    val_set = Dataset(img_path, ori_seg_path, location_path, label_path, 'val')
    val_loader = DataLoader(val_set, batch_size=3, shuffle=False)

    opt = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    sch = StepLR(opt, step_size=20, gamma=0.7)
    loss = DiceLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.float().to(device)

    max_epoch = 151
    cnt = 0
    stop_count = 15
    min_dice_loss = 1.
    stop_flag = False
    for i in range(max_epoch):
        dice_loss_train = epoch_step(train_loader, model, opt, loss, 'train', device)
        dice_loss_val = epoch_step(val_loader, model, opt, loss, 'val', device)
        loss_list = [dice_loss_train, dice_loss_val]
        for j in range(len(loss_list)):
            ws.write(i, j, loss_list[j])

        print(f'in epoch{i}: train dice loss is {dice_loss_train}, val dice loss is {dice_loss_val}')

        if dice_loss_val < min_dice_loss:
            min_dice_loss = dice_loss_val
            save_ckpt(ckpt_path, i, model.state_dict())
            cnt = 0
        else:
            cnt = cnt + 1
        if cnt == stop_count:
            stop_flag = True
            break
        sch.step()

    if not stop_flag:
        save_ckpt(ckpt_path, max_epoch - 1, model.state_dict())
    
    if not os.path.exists(xls_path):
        os.mkdir(xls_path)
    wb.save(os.path.join(xls_path, 'seg_of_rectum_unet.xls'))

if __name__ == "__main__":
    img_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/origin_img_2Dslice'
    ori_seg_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/original_seg_2Dslice'
    location_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/loaction_2Dslice'
    label_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/seg_label_2Dslice'
    ckpt_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/ckpt_with_seg_and_loc'
    xls_path = '/home/zhangqianru/data/seg_of_rectum/div_of_rectum/xls_with_seg_and_loc'

    train(img_path, ori_seg_path, location_path, label_path, ckpt_path, xls_path)
