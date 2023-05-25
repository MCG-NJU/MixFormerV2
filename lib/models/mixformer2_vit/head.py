import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, feat_sz, num_layers, stride, norm=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = feat_sz
        self.img_sz = feat_sz * stride
        if norm:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k), nn.ReLU())
                                          if i < num_layers - 1 
                                          else nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1 
                                          else nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
 
        with torch.no_grad():
            self.indice = torch.arange(0, feat_sz).unsqueeze(0).cuda() * stride # (1, feat_sz)

    def forward(self, reg_tokens, softmax):
        """
        reg_tokens shape: (b, 4, embed_dim)
        """
        reg_token_l, reg_token_r, reg_token_t, reg_token_b = reg_tokens.unbind(dim=1)   # (b, c)
        score_l = self.layers(reg_token_l)
        score_r = self.layers(reg_token_r)
        score_t = self.layers(reg_token_t)
        score_b = self.layers(reg_token_b) # (b, feat_sz)

        prob_l = score_l.softmax(dim=-1)
        prob_r = score_r.softmax(dim=-1)
        prob_t = score_t.softmax(dim=-1)
        prob_b = score_b.softmax(dim=-1)
    
        coord_l = torch.sum((self.indice * prob_l), dim=-1)
        coord_r = torch.sum((self.indice * prob_r), dim=-1)
        coord_t = torch.sum((self.indice * prob_t), dim=-1)
        coord_b = torch.sum((self.indice * prob_b), dim=-1) # (b, ) absolute coordinates

        # return xyxy, ltrb
        if softmax:
            return torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
                prob_l, prob_t, prob_r, prob_b
        else:
            return torch.stack((coord_l, coord_t, coord_r, coord_b), dim=1) / self.img_sz, \
                score_l, score_t, score_r, score_b


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        feat_sz = cfg.MODEL.FEAT_SZ
        stride = cfg.DATA.SEARCH.SIZE / feat_sz
        print("feat size: ", feat_sz, ", stride: ", stride)
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MlpHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            feat_sz=feat_sz,
            num_layers=2,
            stride=stride,
            norm=True
        )
        return mlp_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)


# score
class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg):
    return MlpScoreDecoder(
        in_dim=cfg.MODEL.HIDDEN_DIM,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_layers=2,
        bn=False
    )