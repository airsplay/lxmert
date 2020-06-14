# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
from src.lxrt.modeling import GeLU, BertLayerNorm
from src.lxrt.entry import LXRTEncoder
from param import args


class NLVR2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=20
        )
        self.hid_dim = hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 2)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        :param feat: b, 2, o, f
        :param pos:  b, 2, o, 4
        :param sent: b, (string)
        :param leng: b, (numpy, int)
        :return:
        """
        # Pairing images and sentences:
        # The input of NLVR2 is two images and one sentence. In batch level, they are saved as
        #   [ [img0_0, img0_1], [img1_0, img1_1], ...] and [sent0, sent1, ...]
        # Here, we flat them to
        #   feat/pos = [ img0_0, img0_1, img1_0, img1_1, ...]
        #   sent     = [ sent0,  sent0,  sent1,  sent1,  ...]
        #sent = sum(zip(sent, sent), ())
        batch_size, img_num, obj_num, feat_size = feat.size()
        assert img_num == 1 and obj_num == 36 and feat_size == 2048
        feat = feat.view(batch_size, obj_num, feat_size)
        pos = pos.view(batch_size, obj_num, 4)

        # Extract feature --> Concat
        x = self.lxrt_encoder(sent, (feat, pos))
        x = x.view(-1, self.hid_dim*2)

        # Compute logit of answers
        logit = self.logit_fc(x)

        return logit


