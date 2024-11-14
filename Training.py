import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from datasets_online import SVDD2024
from models import LogSpectrogram
import os
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed, compute_eer_validation
import numpy as np
from torch.utils.data import random_split


def train_epoch(train_loader, loss_function, model, optim, device):
    running_loss = 0
    model.train()
    model.mode='valid'

    for data in train_loader:
        try:
            audio, labels = data[0].to(device), data[1].to(device)
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        with torch.autograd.enable_grad():
            outputs, labels = model(audio, labels)
            loss = loss_function(outputs.squeeze(1), labels.float(), reduction='mean', alpha=-1)    
        running_loss += loss
        optim.zero_grad()
        loss.backward()
        optim.step()

    running_loss /= len(train_loader)
    return running_loss

def evaluate_accuracy(dev_loader, loss_function, model, device):
    running_loss = 0
    model.eval()
    model.mode = 'valid'
    pos_samples = []
    neg_samples = []
    with torch.no_grad():
        for data in dev_loader:
            audio, labels = data[0].to(device), data[1].to(device)
            outputs, labels = model(audio, labels)
            loss = loss_function(outputs.squeeze(1), labels.float(), reduction='mean', alpha=-1)         
            running_loss += loss
            pos_samples.append(outputs[labels == 1].detach().cpu().numpy())
            neg_samples.append(outputs[labels == 0].detach().cpu().numpy())
    val_eer = compute_eer_validation(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]

    running_loss /= len(dev_loader)
    return running_loss, val_eer

def main():
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    lr_max = 1e-4
    wd = 1e-4
    batch_size = 64
    num_workers = 4
    log_dir = 'logs'
    set_seed(8)
    
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('input_type')
    ap.add_argument('experiment_name')
    
    args = ap.parse_args()
    data_dir = Path(args.dataset_dir)
    experiment_name = args.experiment_name
    input_type = args.input_type
    

    model = LogSpectrogram(device)


    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
    
    path = data_dir

    dataset = SVDD2024(path, partition=input_type)
    train_size = int(0.8 * len(dataset))  # 80% for training
    dev_size = len(dataset) - train_size   # 20% for validation
    train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('Number of parameters: ', nb_params)
    
    loss_function = sigmoid_focal_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    best_loss = 1000000000000
    for epoch in range(num_epochs):
        running_loss = train_epoch(train_loader, loss_function, model, optimizer, device)
        valid_loss, valid_eer = evaluate_accuracy(dev_loader, loss_function, model, device)

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Train', running_loss, epoch)
        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        writer.add_scalar('EER/Validation', valid_eer, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)


        print('\nepoch:{} - train_loss:{} - valid_loss:{} - valid_eer:{} - lr:{:.7f}'.format(
            epoch,
            running_loss, valid_loss, valid_eer, optimizer.param_groups[0]['lr']))

        if valid_loss < best_loss:
            print('best model found at epoch: ', epoch)
            torch.save(model.state_dict(), os.path.join(log_dir, '{}_best_epoch.pth'.format(experiment_name, epoch)))
            best_loss = min(valid_loss, best_loss)
        
        scheduler.step()

    writer.close()

if __name__=='__main__':
    main()