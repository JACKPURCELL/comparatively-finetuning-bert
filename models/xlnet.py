from transformers import XLNetConfig, XLNetModel



#!/usr/bin/env python3

import torch

import torchvision.models.vision_transformer as XLNet
from transformers import AutoTokenizer, XLNetForSequenceClassification

import torch.nn as nn
from collections import OrderedDict
from collections.abc import Callable


class XLNet(nn.Module):

    def __init__(self, name: str = 'xlnet-base-cased', num_classes=2, parallel = True, **kwargs):
        super(XLNet, self).__init__()

        if parallel:
            self.model = nn.DataParallel(XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_classes)).cuda()
        else:
            self.model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_classes).cuda()

        

    def forward(self, input_ids,token_type_ids,attention_mask):

        
        return self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask).logits

