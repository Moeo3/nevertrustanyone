from dice_loss import DiceLoss
from unet import UNet
from emb_dataset import EmbDataset
import torch
from torch.optim import Adam
import os
from torch.utils.data import DataLoader

# def re_train(net, dataloader, save_path, epoch=6):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net = net.float().to(device)
#     net.train()
#     opt = Adam(net.parameters(), lr=1e-4)
#     loss = DiceLoss()

#     for i in range(epoch):
#         for step, batch in enumerate(dataloader):
#             feature = batch['features'].to(device)
#             label = batch['labels'].to(device)
#             pred = net(feature)
#             dice_loss = loss(label, pred)
#             print(f'Dice loss is {dice_loss} in epoch {i} step {step}')

#             opt.zero_grad()
#             dice_loss.backward()
#             opt.step()   
#             pass

#         state = {
#             'net' : net.state_dict(),
#             'opt' : opt.state_dict(),
#             'epoch' : i
#         }
#         try:
#             os.mkdir(save_path)
#         except:
#             pass
#         torch.save(state, os.path.join(save_path, f'epoch{i}.pth'))
#         pass
#     pass

def re_train(net, dataloader, )

if __name__ == "__main__":
    
    # dataset = EmbDataset()
    # channels_in = len(dataset.model_set) + 1
    # re_train(UNet(channels_in, 1), DataLoader(dataset, batch_size=3, shuffle=True))
    pass
