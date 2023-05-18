import torch
import torchvision
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, channels_img, feature_d):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(                                                    # The structure of D
            # input: n * channels_img * 64 * 64
            nn.Conv2d(channels_img, feature_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(feature_d, feature_d*2, 4, 2, 1),                             # 16*16
            self._block(feature_d * 2, feature_d * 4, 4, 2, 1),                       # 8*8
            self._block(feature_d * 4, feature_d * 8, 4, 2, 1),                       # 4*4

            nn.Conv2d(feature_d * 8, 1, kernel_size=4, stride=2, padding=0),          # 1*1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.D(x)



class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()                                             # The structure of G
        self.G = nn.Sequential(
            #　input: n*z_dim*1*1
            self._block(z_dim, features_g*16, 4, 1, 0),                               # n * f_g * 16 4 * 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),                    # 8 * 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),                     # 16 * 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),                     # 32 * 32
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()                                                                 # [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.G(x)



def initialize_weights(model):
    for m in model.modules():

        # if isinstance(m, nn.Conv2d):
        #     nn.init.normal_(m.weight.data, 0.0, 0.02)
        # if isinstance(m, nn.ConvTranspose2d):
        #     nn.init.normal_(m.weight.data, 0.0, 0.02)
        # if isinstance(m, nn.BatchNorm2d):
        #     nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



def test():
    N, in_channels, H, W = 8, 3, 64, 64                            # 定义输入的图像数据是 8张，三通道的（RGB），64*64的影像
    z_dim = 100                                                    # 噪声影像的维度为100
    x = torch.randn((N, in_channels, H, W))                        # 生成和定义影像尺寸大小相等的张量

    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)


    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)

    print("Success!")


test()



