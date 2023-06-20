from model.module import CBT, AMF
from os.path import join as pjoin
from os.path import exists as pexists
from tqdm import tqdm

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
import random


class baseModel_best(nn.Module):
    def __init__(self, modelConfig):
        super().__init__()

        num_classes, frames = modelConfig['num_classes_frames']
        linear_dropout, temporal_dropout, fusion_dropout = modelConfig['linear_temporal_fusion_dropout']
        dim, hidden_dim = modelConfig['dim_hiddenDim']
        depth, heads = modelConfig['depth_heads']

        self.vision_linear = nn.Sequential(nn.LayerNorm(3584), nn.Linear(3584, dim), nn.ReLU(inplace=True), nn.Dropout(linear_dropout))
        self.audio_linear = nn.Sequential(nn.LayerNorm(1583), nn.Linear(1583, dim), nn.ReLU(inplace=True), nn.Dropout(linear_dropout))
        self.vggish_linear = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, dim), nn.ReLU(inplace=True), nn.Dropout(linear_dropout))
        self.scene_linear = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, dim), nn.ReLU(inplace=True), nn.Dropout(linear_dropout))
        self.face_linear = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, dim), nn.ReLU(inplace=True), nn.Dropout(linear_dropout))
        self.video_linear = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, dim), nn.ReLU(inplace=True), nn.Dropout(linear_dropout))

        self.pos_embedding = nn.Parameter(torch.randn(1, frames + 1, dim))

        self.vision_temporal = nn.Sequential(CBT(dim, depth, heads, hidden_dim, temporal_dropout))
        self.audio_temporal = nn.Sequential(CBT(dim, depth, heads, hidden_dim, temporal_dropout))
        self.vggish_temporal = nn.Sequential(CBT(dim, depth, heads, hidden_dim, temporal_dropout))
        self.scene_temporal = nn.Sequential(CBT(dim, depth, heads, hidden_dim, temporal_dropout))
        self.face_temporal = nn.Sequential(CBT(dim, depth, heads, hidden_dim, temporal_dropout))

        self.fusion = AMF(dim, 1, heads, hidden_dim, fusion_dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, images_data, audios_data, vggish_data, scenes_data, face_data, video_data, prev_label):
        b, t, _ = images_data.shape
        images_data = self.vision_linear(images_data)
        audios_data = self.audio_linear(audios_data)
        vggish_data = self.vggish_linear(vggish_data)
        scenes_data = self.scene_linear(scenes_data)
        face_data = self.face_linear(face_data)

        video_data = self.video_linear(video_data)

        images_data = torch.cat((video_data, images_data), 1)
        audios_data = torch.cat((video_data, audios_data), 1)
        vggish_data = torch.cat((video_data, vggish_data), 1)
        scenes_data = torch.cat((video_data, scenes_data), 1)
        face_data = torch.cat((video_data, face_data), 1)

        images_data = self.vision_temporal(images_data)
        audios_data = self.audio_temporal(audios_data)
        vggish_data = self.vggish_temporal(vggish_data)
        scenes_data = self.scene_temporal(scenes_data)
        face_data = self.face_temporal(face_data)

        x = torch.stack([images_data[:, 1:, ], audios_data[:, 1:, ], vggish_data[:, 1:, ], scenes_data[:, 1:, ], face_data[:, 1:, ]], 2)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.fusion(x)
        x = x.mean(1)
        x = rearrange(x, '(b t) d -> b t d', b=b, t=t)
        x = self.classifier(x)
        return x
