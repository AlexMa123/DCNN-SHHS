import torch
from torch import nn


def padding_same(kernel_size=3, dilation=1, stride=1):
    # stride * Lout = Lin + 2padding - dilation (kernel_size - 1) - 1 + stride
    # (stride - 1) * Lout + dilation (kernel_size - 1) + 1 - stride = 2 padding
    padding = dilation * (kernel_size - 1) + 1 - stride
    return padding // 2


class DCNN_classifier(nn.Module):
    """DCNN network for sleep stage classification

    """
    def __init__(self,
                 epoch_length=(512, 128),
                 num_windows_features=128,
                 num_channels_rri=[8, 16, 32, 64],
                 num_channels_mad=[8, 16, 32, 64],
                 dilations=[2, 4, 8, 16, 32],
                 n_classes=4):
        """
        DCNN network for sleep stage classification based on RRI and MAD

        Parameters
        ----------
        epoch_length : tuple, optional
            length of the input signals in one epoch.
            For example, 128s RRI (4Hz) and 128s MAD (1Hz) will be (512, 128), by default (512, 128)
        num_windows_features : int, optional
            how many features learned from each window, by default 128
        num_channels_rri : list, optional
            number of channels to process RRI signal by 1d-convolution, by default [8, 16, 32, 64]
        num_channels_mad : list, optional
            number of channels to process MAD signal, by default [8, 16, 32, 64]
        dilations : list, optional
            for DCNN temporal network, by default [2, 4, 8, 16, 32]
        n_classes : int, optional
            number of predict sleep stages., by default 4
        """
        super(DCNN_classifier, self).__init__()
        self.w1, self.w2 = epoch_length
        self.num_pooling_rri = len(num_channels_rri) - 1
        self.num_pooling_mad = len(num_channels_mad) - 1
        # ======================================================================================
        # Windows learning part
        # signal feature learning part
        # RRI
        self.input_convolution_rri = nn.Conv1d(1, num_channels_rri[0], 1)
        cnn_layers = []
        cnn_layers.append(nn.LeakyReLU(0.15))
        cnn_layers.append(nn.BatchNorm1d(num_channels_rri[0]))

        for i in range(0, len(num_channels_rri) - 1):
            cnn_layers.append(nn.Conv1d(num_channels_rri[i],
                                        num_channels_rri[i + 1],
                                        3, padding=padding_same(3)))
            cnn_layers.append(nn.MaxPool1d(2),)
            cnn_layers.append(nn.LeakyReLU(0.15))
            cnn_layers.append(nn.BatchNorm1d(num_channels_rri[i+1]))

        self.signallearning_rri = nn.Sequential(*cnn_layers)
        # MAD
        self.input_convolution_mad = nn.Conv1d(1, num_channels_mad[0], 1)
        cnn_layers = []
        cnn_layers.append(nn.LeakyReLU(0.15))
        cnn_layers.append(nn.BatchNorm1d(num_channels_mad[0]))

        for i in range(0, len(num_channels_mad) - 1):
            cnn_layers.append(nn.Conv1d(num_channels_mad[i],
                                        num_channels_mad[i + 1],
                                        3, padding=padding_same(3)))
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.LeakyReLU(0.15))
            cnn_layers.append(nn.BatchNorm1d(num_channels_mad[i+1]))
        self.signallearning_mad = nn.Sequential(*cnn_layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((epoch_length[0] // (2 ** self.num_pooling_rri) * num_channels_rri[-1] +
                                 epoch_length[1] // (2 ** self.num_pooling_mad) * num_channels_mad[-1]),
                                num_windows_features)
        # Windows feature learning END
        # =====================================================================================
        self.classfier = nn.Sequential(
            ResBlock(num_windows_features, 7, dilations),
            ResBlock(num_windows_features, 7, dilations),
            nn.Conv1d(num_windows_features, n_classes, 1)
        )

    def forward(self, rri, mad=None):
        batchsize, Nwindows, w1 = rri.shape
        assert w1 == self.w1, 'the input length of RRI is not the same with your configuration of NN'
        rri = rri.reshape(-1, 1, w1)
        out = self.input_convolution_rri(rri)
        out = self.signallearning_rri(out)
        if mad is not None:
            _, _, w2 = mad.shape
            assert w2 == self.w2, 'the input length of MAD is not the same with your configuration of NN'
            mad = mad.reshape(-1, 1, w2)
            out_mad = self.input_convolution_mad(mad)
            out_mad = self.signallearning_mad(out_mad)
        else:
            out_mad = torch.zeros((out.shape[0], out.shape[1],
                                   self.w2 // (2 ** self.num_pooling_mad)), device=rri.device)
        out = torch.cat([out, out_mad], dim=-1)

        out = self.flatten(out)
        out = self.linear(out)
        out = out.reshape(batchsize, Nwindows, -1)
        out = out.transpose(1, 2).contiguous()
        out = self.classfier(out)
        out = out.transpose(1, 2).contiguous().reshape(batchsize * Nwindows, -1)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_windows_features=128, kernel_size=7,
                 dilations=[2, 4, 8, 16, 32]):
        super(ResBlock, self).__init__()
        cnn_list = []
        for d in dilations:
            cnn_list.append(nn.LeakyReLU(0.15))
            cnn_list.append(nn.Conv1d(
                num_windows_features, num_windows_features,
                kernel_size=kernel_size,
                dilation=d,
                padding=padding_same(kernel_size, d)
            ))
            cnn_list.append(nn.Dropout(0.2))
        self.cnn = nn.Sequential(
            *cnn_list
        )

    def forward(self, x):
        out = self.cnn(x)
        out = x + out
        return out


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    test_rri = torch.rand((1, 1200, 512), device=device)
    test_mad = torch.rand((1, 1200, 128), device=device)

    mynet = DCNN_classifier(num_channels_rri=[2, 4, 8, 16, 32, 64], dilations=[2, 4, 6, 8]).to(device)
    out = mynet(test_rri, test_mad)
    print(out.shape)
