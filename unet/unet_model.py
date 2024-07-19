""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """
        use_checkpointing 方法将模型各层的前向传播过程设置为使用 PyTorch 的检查点机制。
        这可以在训练大型模型时减少显存占用,但会增加训练时间。
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

"""
根据这个 UNet 模型的定义,我们可以得到以下关于输入和输出数据的信息:

输入数据大小和通道数:
输入数据的通道数由 n_channels 参数指定。这意味着输入图像可以是单通道(如灰度图像)或多通道(如 RGB 图像)。
输入数据的空间大小没有明确指定,因为 UNet 模型可以处理不同大小的输入。但通常情况下,输入图像的大小应该足够大,以便于模型在编码器-解码器结构中进行多尺度特征提取。

输出数据大小和通道数:
输出数据的通道数由 n_classes 参数指定。这意味着输出图像的通道数对应于分类任务的类别数。
与输入类似,输出数据的空间大小也没有明确指定。但通常情况下,输出图像的大小应该与输入图像的大小相同,以便于进行逐像素的分类。
举个例子,如果我们设置 n_channels=3 和 n_classes=2,那么:
输入数据应该是一个 3 通道的图像,例如 RGB 图像。
输出数据将是一个 2 通道的图像,每个通道对应一个分类类别。

总之,UNet 模型的输入和输出大小和通道数是可配置的,根据具体的任务需求而定。关键是要确保输入和输出的空间大小和通道数与模型的参数设置相匹配。
"""