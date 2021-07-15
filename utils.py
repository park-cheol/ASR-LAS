import math

import torch
import torch.nn as nn

class MaskConv(nn.Module):

    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, input, lengths):
        """
        Adds padding to the output of the module based on the given lengths..
        This is to ensure that
        the results of the model do not change when batch sizes change during inference.
        :param input: shape [B, C, D, T] # todo D, T의 의미
        :param lengths: batch에서 각각의 sequence 실제 길이
        :return: module 로부터 masked output
        """
        # todo 실제 인자들이 어떻지 확인
        for module in self.seq_module:
            input = module(input)
            mask = torch.BoolTensor(input.size()).fill_(0) # 같은 사이즈로 모두 False로 선언

            if input.is_cuda: # cuda 인지 확인
                mask = mask.cuda() # mask 도 gpu로 올려줌

            # todo 동작 확인
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
                    # torch.narrow(input, dim, start, length) → Tensor
                    # input 의 좁아진 verson을 반환
                    # dim: narrow 할 dim , start 부터 start + length(포함x) 까지 출력

            input = input.masked_fill(mask, 0)
            # masked_fill(mask, value): mask는 bool Tensor, value: 채울 값

        return input, lengths

