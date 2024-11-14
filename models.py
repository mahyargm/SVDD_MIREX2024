import torch
from torchvision.models import resnet18
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchaudio.transforms as T


def modify_for_grayscale(model):

    first_conv_layer = model.conv1 # resnet18

    new_first_conv_layer = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )
    # Copy the weights from the original convolutional layer to the new one
    with torch.no_grad():
        new_first_conv_layer.weight[:, 0] = first_conv_layer.weight.mean(dim=1)
        if first_conv_layer.bias is not None:
            new_first_conv_layer.bias = first_conv_layer.bias
    # Replace the first convolutional layer in the model
    model.conv1 = new_first_conv_layer # resnet18
    return model


class LogSpectrogram(nn.Module):
    def __init__(self, device, sample_rate=44100, n_fft= 2048, win_length= int(0.025*44100), hop_length= int(0.01*44100)):
        super(LogSpectrogram, self).__init__()
        self.device = device
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        self.stft = T.Spectrogram(n_fft=self.n_fft,
                                  win_length=self.win_length,
                                  hop_length=self.hop_length,
                                  power=2).to(device)


        self.model = resnet18(pretrained=True)
        self.model = modify_for_grayscale(self.model)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 256, bias=True, device=device),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.model.to(device)

        self.to_db = T.AmplitudeToDB()


        self.normalize = torchvision.transforms.Normalize(mean=0.449,std=0.226)

    def forward(self, x, labels):
        x = x.unsqueeze(1)
        x = self.stft(x)
        x = self.to_db(x)
        x = self.normalize(x)
        labels = labels*1.
        x = self.model(x)
        return x, labels

