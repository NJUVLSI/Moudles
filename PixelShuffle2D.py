'''
同torch中的pixelshuffle有一样的功能，参考自https://github.com/assassint2017/PixelShuffle3D

'''

import torch.nn as nn


class PixelShuffle2D(nn.Module):
    
    def __init__(self, upscale_factor):
       
        super(PixelShuffle2D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 2

        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor,
            in_height, in_width)
        print(input_view.size())

        shuffle_out = input_view.permute(0,1,4,2,5,3).contiguous()

        return shuffle_out.view(batch_size, channels, out_height, out_width)




