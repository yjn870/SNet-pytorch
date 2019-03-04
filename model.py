from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionalUnit(nn.Module):
    def __init__(self, structure_type):
        super(ConvolutionalUnit, self).__init__()
        self.structure_type = structure_type

        if structure_type == 'classic':
            self.net = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        elif structure_type == 'advanced':
            self.net = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(structure_type)

    def forward(self, x):
        residual = x
        x = self.net(x)
        if self.structure_type == 'advanced':
            x = 0.1 * x
        x = residual + x
        return x


class S_Net(nn.Module):
    def __init__(self, num_metrics=8, structure_type='classic'):
        super(S_Net, self).__init__()
        self.num_metrics = num_metrics

        self.encoder = EncoderBlock()
        self.convolution_units = nn.Sequential(*[ConvolutionalUnit(structure_type) for i in range(num_metrics)])
        self.decoders = nn.Sequential(*[DecoderBlock() for i in range(num_metrics)])

    def forward(self, x):
        x = self.encoder(x)

        outs = []
        prev_out = x
        for i in range(self.num_metrics):
            out = self.convolution_units[i](prev_out)
            prev_out = out
            outs.append(self.decoders[i](out))

        return outs
