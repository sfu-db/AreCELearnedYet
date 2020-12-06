import torch.nn as nn

class LWNNLayer(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        return self.layer(X)

class LWNNModel(nn.Module):
    def __init__(self, input_len, hid_units):
        super().__init__()
        self.hid_units = hid_units

        self.hid_layers = nn.Sequential()
        for l, output_len in enumerate([int(u) for u in hid_units.split('_')]):
            self.hid_layers.add_module('layer_{}'.format(l), LWNNLayer(input_len, output_len))
            input_len = output_len

        self.final = nn.Linear(input_len, 1)

    def forward(self, X):
        mid_out = self.hid_layers(X)
        pred = self.final(mid_out)

        return pred

    def name(self):
        return f"lwnn_hid{self.hid_units}"
