import torch
import os
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.models import resnet18
import torch.nn as nn
import argparse
from datasets_online import SVDD2024
import torchaudio
from utils import compute_eer
from models import LogSpectrogram
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def prediction(model, dataloader, device):
    model.eval()
    outputs = []
    labels = []
    file_names = []
    with torch.no_grad():
        for batch in dataloader:
            audio, targets, file_name = batch
            audio = audio.to(device)
            output, targets = model(audio, targets)
            outputs.append(output.cpu().numpy())
            labels.append(targets)
            file_names.append(file_name)
            
    outputs = np.vstack(outputs)
    labels = np.concatenate(labels)
    file_names = np.concatenate(file_names)
    return outputs, labels, file_names


            
        
def main():

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    num_workers = 2
    log_dir = 'logs'


    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('input_type')
    ap.add_argument('experiment_name')
    

    args = ap.parse_args()
    data_dir = Path(args.dataset_dir)
    experiment_name = args.experiment_name
    input_type = args.input_type

    

    model = LogSpectrogram(device)

    model.load_state_dict(torch.load(os.path.join(log_dir, f'{experiment_name}_best_epoch.pth')))
    model.eval()
    print(f"Experiment name: {experiment_name}")
    # Create the dataset
    path = data_dir

    test_dataset = SVDD2024(path, partition=input_type, test_flag=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)


    print('Test prediction started!')
    test_predictions, test_labels, test_file_names = prediction(model, test_loader, device)

    # Calculte scores for each segment
    segment_names_dict = defaultdict(list)
    for index, file_name in enumerate(test_file_names):
        segment_name = '_'.join(file_name.split('_')[:-1]).strip()
        segment_names_dict[segment_name].append((test_labels[index], test_predictions[index]))
        
    test_labels = []
    test_predictions = []
    for segment in segment_names_dict.keys():
        l = int(segment_names_dict[segment][0][0])
        p = sum([float(item[1]) for item in segment_names_dict[segment]]) / len(segment_names_dict[segment])
        s_name = '_'.join(segment.split('_')[:-1]).strip()
        s_number = segment.split('_')[-1].strip()
        test_labels.append(l)
        test_predictions.append(p)
        with open(os.path.join('./scores', f'scores_{experiment_name}_{input_type}.txt'), "a") as f:
            f.write(f"{s_name} {s_number} {format(p, '.16f')} {l}\n")

        
    test_predictions = np.array(test_predictions)
    test_labels = np.array(test_labels)
    eer = compute_eer(test_labels, test_predictions)
    auc = roc_auc_score(test_labels, test_predictions)
    print(f'The EER for the test dataset is: ', eer[0])
    print(f'The AUC for the test dataset is: ', auc)
    print('Done!')


if __name__=='__main__':
    main()