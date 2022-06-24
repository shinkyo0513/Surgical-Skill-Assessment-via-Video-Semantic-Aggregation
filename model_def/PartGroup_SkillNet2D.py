import torch
from torch import nn
from torchvision import models

import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_def.PartGroupNet import PartGroupNet, BasicBlock1x1
from model_def.GCN import GCN
from model_def.Multi_LSTMs import Multi_LSTMs

from vit_pytorch.vit import Transformer
from utils.PositionalEncoding import PositionalEncoding

class PartGroup_SkillNet2D (nn.Module):
    def __init__ (self, num_parts=3, extractor_type='r2d', context_type='none', aggregate_type='mean',
                  avgpool_parts=False, scene_node=False, attention=False, multi_lstms=False, 
                  prepro=False, no_pastpro=False, simple_pastpro=False, 
                  final_score_weight=25, final_score_bias=15):
        super(PartGroup_SkillNet2D, self).__init__()
        self.scene_node = scene_node
        self.avgpool_parts = avgpool_parts

        self.num_parts = max(num_parts, 1)
        # self.num_nodes = self.num_parts + 1 if self.scene_node else self.num_parts
        self.num_nodes = self.num_parts if not self.avgpool_parts else 1
        # self.num_nodes = self.num_nodes + 1 if self.scene_node else self.num_nodes

        self.extractor_type = extractor_type
        self.context_type = context_type
        self.aggregate_type = aggregate_type

        self.attention = attention
        self.no_pastpro = no_pastpro
        self.prepro = prepro
        self.simple_pastpro = simple_pastpro
        self.multi_lstms = multi_lstms

        self.final_score_weight = final_score_weight
        self.final_score_bias = final_score_bias

        # 2D: ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
        # maxpool is removed for feature maps with larger sizes
        if extractor_type == 'r2d':
            self.extractor = models.resnet18(pretrained=True)
            # 2D: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            self.featmap_extractor = nn.Sequential(*(list(self.extractor.children())[0:3]+list(self.extractor.children())[4:-3]))   # -3, layer3, 256x14x14
            self.num_chnals = self.featmap_extractor[-1][-1].bn2.num_features
        if extractor_type == 'r2d_4layer':
            self.extractor = models.resnet18(pretrained=True)
            # 2D: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            self.featmap_extractor = nn.Sequential(*(list(self.extractor.children())[0:3]+list(self.extractor.children())[4:-2]))   # -2, layer4, 512x7x7
            self.num_chnals = self.featmap_extractor[-1][-1].bn2.num_features
        elif extractor_type == 'r2d_50':
            self.extractor = models.resnet50(pretrained=True)
            # 2D: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            self.featmap_extractor = nn.Sequential(*(list(self.extractor.children())[0:3]+list(self.extractor.children())[4:-3]))
            self.num_chnals = self.featmap_extractor[-1][-1].bn3.num_features
        elif extractor_type == 'r2d_101':
            self.extractor = models.resnet101(pretrained=True)
            # 2D: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            self.featmap_extractor = nn.Sequential(*(list(self.extractor.children())[0:3]+list(self.extractor.children())[4:-3]))
            self.num_chnals = self.featmap_extractor[-1][-1].bn3.num_features

        if self.num_parts > 1:
            self.part_grouping = PartGroupNet(self.num_parts, self.num_chnals)
            self.part_grouping.init_parameters(init_centers=None, init_smooth_factors=None)

        # post-processing bottleneck block for the region features
        self.part_out_chnals = self.num_chnals if self.no_pastpro else 256
        part_out_downsample = nn.Sequential(
            nn.Conv2d(self.num_chnals, self.part_out_chnals, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.part_out_chnals)
        ) if self.part_out_chnals != self.num_chnals else None
        if self.simple_pastpro:
            self.part_postpro = BasicBlock1x1(self.num_chnals, self.part_out_chnals, stride=1, downsample=part_out_downsample)
        else:
            self.part_postpro = nn.Sequential(
                BasicBlock1x1(self.num_chnals, self.part_out_chnals, stride=1, downsample=part_out_downsample),
                nn.Conv2d(self.part_out_chnals, self.part_out_chnals, kernel_size=1, stride=1)
            )
        part_modules = list(self.part_postpro.modules())

        if self.prepro:
            self.part_prepro = BasicBlock1x1(self.num_chnals, self.num_chnals, stride=1, downsample=None)
            part_modules += list(self.part_prepro.modules())

        if self.scene_node:
            self.scene_postpro = nn.Sequential(
                BasicBlock1x1(self.num_chnals, self.part_out_chnals, stride=1, downsample=part_out_downsample),
                nn.Conv2d(self.part_out_chnals, self.part_out_chnals, kernel_size=1, stride=1)
            )
            part_modules += list(self.scene_postpro.modules())

        for m in part_modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, BasicBlock1x1):
                nn.init.constant_(m.bn2.weight, 0)

        if context_type == 'lstm':
            self.context_out_size = 128
            self.rnn = Multi_LSTMs(num_lstms=self.num_nodes,
                                   num_input_chnals=self.part_out_chnals,
                                   num_output_chanls=self.context_out_size,
                                   bidirectional=False,
                                   detach_hidden_state=False,
                                   share_param= (not self.multi_lstms) )
            if self.scene_node:
                self.scene_rnn = Multi_LSTMs(num_lstms=1,
                                   num_input_chnals=self.part_out_chnals,
                                   num_output_chanls=self.context_out_size,
                                   bidirectional=False,
                                   detach_hidden_state=False,
                                   share_param=False)
        elif context_type == 'bilstm':
            self.context_out_size = 256
            self.rnn = Multi_LSTMs(num_lstms=self.num_nodes, 
                                   num_input_chnals=self.part_out_chnals, 
                                   num_output_chanls=self.context_out_size, 
                                   bidirectional=True, 
                                   detach_hidden_state=True,
                                   share_param=(not self.multi_lstms) )
            if self.scene_node:
                self.scene_rnn = Multi_LSTMs(num_lstms=1,
                                   num_input_chnals=self.part_out_chnals,
                                   num_output_chanls=self.context_out_size,
                                   bidirectional=True,
                                   detach_hidden_state=True,
                                   share_param=False)
        elif context_type == 'transformer':
            self.context_out_size = self.part_out_chnals
            # depth * {LN + MHA + LN + MLP}
            self.transformer = Transformer(self.part_out_chnals, depth=2, heads=4, dim_head=64, mlp_dim=32, dropout=0.2)
            if self.scene_node:
                self.scene_transformer = Transformer(self.part_out_chnals, depth=2, heads=4, dim_head=64, mlp_dim=32, dropout=0.2)
            self.pe = PositionalEncoding(self.part_out_chnals, dropout=0.2)
        elif context_type == 'gcn':
            self.context_out_size = 256
            self.gcn_mid_size = self.context_out_size // 4
            self.gcn = GCN(self.part_out_chnals, self.gcn_mid_size, self.context_out_size)
        elif context_type == 'none':
            self.context_out_size = self.part_out_chnals

        if self.scene_node:
            self.aggregate_in_size = self.context_out_size * (self.num_nodes + 1)
        else:
            self.aggregate_in_size = self.context_out_size * self.num_nodes
        self.aggregate_mid_size = self.aggregate_in_size // 4
        if aggregate_type == 'mean':
            self.aggregate = nn.Sequential(
                                nn.Conv1d(self.aggregate_in_size, 1, kernel_size=1, stride=1, padding=0),
                                nn.Tanh(),
                                nn.AdaptiveAvgPool1d(1)
                            )
        elif aggregate_type == 'avgpool':
            self.aggregate = nn.Sequential(
                                nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(self.aggregate_in_size, 1, kernel_size=1, stride=1, padding=0),
                                nn.Tanh()
                            )
        elif aggregate_type == 'final':
            self.aggregate = nn.Sequential(
                                nn.Linear(self.aggregate_in_size, 1, bias=True),
                                nn.Tanh()
                            )
        elif aggregate_type == 'lstm':
            self.aggregate_lstm = nn.LSTM(self.aggregate_in_size, self.aggregate_mid_size, batch_first=True)
            self.aggregate_fc = nn.Sequential(
                                nn.Linear(self.aggregate_mid_size, 1, bias=True),
                                nn.Tanh()
                            )

    def forward (self, clip_tensor):
        # Input: clip_tensor BxLxCxHxW
        # print(clip_tensor.shape)
        bs = clip_tensor.shape[0]
        device = clip_tensor.device

        tensors = clip_tensor.unbind(dim=1) # list of BxCxHxW, length L
        nt = len(tensors)
        tensors = torch.cat(tensors, dim=0).contiguous() # L*B x CxHxW
        featmaps = self.featmap_extractor(tensors)     # L*B x256x7x7, 
        
        if self.prepro:
            featmaps = self.part_prepro(featmaps)
        featmaps_mean = featmaps.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)    # L'*B x 256x1x1
        assert featmaps.shape[1] == self.num_chnals

        if self.num_parts > 1:
            part_feats, part_assigns = self.part_grouping(featmaps) # L'*B x256x3, L'*B x3x14x14
            part_feats = part_feats.contiguous().unsqueeze(-1)  # L'*B x256x3x1
        else:
            part_feats = featmaps_mean  # L'*B x 256x1x1
            part_assigns = torch.ones((nt*bs, 3, 14, 14)).to(device)     # L'*B x3x14x14
        
        if not self.no_pastpro:
            part_feats = self.part_postpro(part_feats)          # L'*B x256x3x1

        if self.avgpool_parts:
            part_feats = part_feats.mean(dim=2, keepdim=True) # L'*B x256x1x1

        part_feats = part_feats.squeeze(3)      # L'*B x256x3
        part_feats = torch.stack(torch.split(part_feats, bs, dim=0), dim=1).contiguous()    # BxL'x256x3, R, nomalized

        if self.context_type == 'lstm' or self.context_type == 'bilstm':
            rnn_out = self.rnn(part_feats)  # BxL'x128x3
            if self.scene_node: 
                scene_feat = self.scene_postpro(featmaps_mean).squeeze(3)   # L'*B x256x1
                scene_feat = torch.stack(torch.split(scene_feat, bs, dim=0), dim=1).contiguous()    # BxL'x256x1
                scene_rnn_out = self.scene_rnn(scene_feat)  # BxL'x128x1
                rnn_out = torch.cat([rnn_out, scene_rnn_out], dim=3)    # BxL'x128x4
            context_out = torch.cat(torch.unbind(rnn_out, dim=3), dim=2)    # Bx L' x4*128
            context_out = context_out.transpose(1, 2).contiguous()  # Bx 4*128 xL'
        elif self.context_type == 'transformer':
            transformer_out = []
            for pi in range(self.num_parts):
                transformer_out.append(self.transformer(self.pe(part_feats[:,:,:,pi])))
            if self.scene_node:
                scene_feat = self.scene_postpro(featmaps_mean).squeeze(3)   # L'*B x256x1
                scene_feat = torch.stack(torch.split(scene_feat, bs, dim=0), dim=1).squeeze(-1).contiguous()    # BxL'x256
                transformer_out.append(self.scene_transformer(self.pe(scene_feat)))
            context_out = torch.cat(transformer_out, dim=2).transpose(1, 2) # B x 4*128 x L'
        elif self.context_type == 'gcn':
            gcn_in = torch.cat(torch.unbind(part_feats, dim=1), dim=2)    # Bx256x L'*3
            gcn_in = gcn_in.transpose(1,2).contiguous()  # Bx L'*3 x 256
            k = self.num_nodes
            part_adj = torch.eye(nt*k).to(device)   # 4*3 x 4*3
            for ridx in range(part_adj.shape[0]):
                for cidx in range(part_adj.shape[1]):
                    # idx//k == timestep idx, idx%k == part class idx
                    if cidx//k == ridx//k:  # For parts in the same timestep
                        part_adj[ridx, cidx] = 1.0
                    else:                   # For parts in different timesteps
                        if cidx%k == ridx%k:    # If two parts belonging to the same class
                            dist = abs(float(cidx//k - ridx//k))
                            part_adj[ridx, cidx] = 1.0 - dist / nt
                        else:
                            part_adj[ridx, cidx] = 0.0
            part_adj = part_adj.unsqueeze(0).expand(bs, -1, -1)    # B x L'*3 x L'*3
            gcn_out = self.gcn(gcn_in, part_adj).contiguous()    # B x L'*3 x256
            gcn_out = torch.stack(torch.split(gcn_out, k, dim=1), dim=-1).contiguous() # B x 3 x 256 x L'
            context_out = torch.cat(torch.unbind(gcn_out, dim=1), dim=1)  # B x 3*256 x L'
        elif self.context_type == 'none':
            part_feats = nn.functional.relu(part_feats)
            context_out = torch.cat(part_feats.unbind(dim=3), dim=2)    # BxL'x 3*256
            context_out = context_out.transpose(1,2).contiguous()    # Bx 3*256 xL'

        if self.aggregate_type == 'mean' or self.aggregate_type == 'avgpool':
            out = self.aggregate(context_out) # B x 1 x 1
            out = out.squeeze(2)    # B x 1
        elif self.aggregate_type == 'final':
            out = self.aggregate(context_out[:,:,-1])   # B x 1
        elif self.aggregate_type == 'lstm':
            out = self.aggregate_lstm(context_out.transpose(1,2))[0]   # B x L' x 1
            out = self.aggregate_fc(out)   # B x L' x 1
            out = out[:, -1, :]     # B x 1
        final_score = self.final_score_weight * out + self.final_score_bias     # B, -10 ~ 40
        
        part_assigns = torch.stack(torch.split(part_assigns, bs, dim=0), dim=1)     # B x L x 3x14x14
        if not self.attention:
            part_att = torch.ones((bs, nt, self.num_parts)).to(device)   # BxLx3
        
        return final_score, part_assigns, part_att

