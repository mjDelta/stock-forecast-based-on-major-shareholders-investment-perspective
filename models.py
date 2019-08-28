#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-23 14:16:07
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
from torch import nn
import torch
from layers import GraphConvolution
import torch.nn.functional as F

class NetPro2(nn.Module):
	def __init__(self,hid_dim,code_size,owner_size):
		super(NetPro2,self).__init__()
		self.code_emb_layer=nn.Embedding(code_size,hid_dim)
		self.owner_emb_layer=nn.Embedding(owner_size,hid_dim)
		self.mix_layer1=nn.Sequential(
			nn.Linear(2*hid_dim+2,hid_dim),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.mix_layer2=nn.Sequential(
			nn.Linear(hid_dim,hid_dim),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.mix_layer3=nn.Sequential(
			nn.Linear(hid_dim*3,2),
			nn.Sigmoid())
	def forward(self,codes,owners,others):
		code_emb=self.code_emb_layer(codes.to(torch.int64))
		owner_emb=self.owner_emb_layer(owners.to(torch.int64))
		mix=torch.cat([code_emb,owner_emb,others],dim=1)
		mix=self.mix_layer1(mix)
		mix=self.mix_layer2(mix)
		tmp=torch.cat([mix,code_emb,owner_emb],dim=1)
		tmp=self.mix_layer3(tmp)
		return tmp
