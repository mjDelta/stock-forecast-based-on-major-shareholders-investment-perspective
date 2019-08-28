#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-23 14:16:07
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
from torch import nn
import torch
from layers import GraphConvolution
import torch.nn.functional as F
class Net(nn.Module):
	def __init__(self,hid_dim,code_size,owner_size):
		super(Net,self).__init__()
		self.code_emb_layer=nn.Embedding(code_size,hid_dim)
		self.owner_emb_layer=nn.Embedding(owner_size,hid_dim)
		self.hidden_layer1=nn.Sequential(
			nn.Linear(2*hid_dim+2,hid_dim),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.hidden_layer2=nn.Sequential(
			nn.Linear(hid_dim,hid_dim),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.hidden_layer3=nn.Sequential(
			nn.Linear(hid_dim,2),
			nn.Sigmoid())
	def forward(self,codes,owners,others):
		code_emb=self.code_emb_layer(codes.to(torch.int64))
		owner_emb=self.owner_emb_layer(owners.to(torch.int64))

		tmp=torch.cat([code_emb,owner_emb,others],dim=1)
		tmp=self.hidden_layer1(tmp)
		tmp=self.hidden_layer2(tmp)
		tmp=self.hidden_layer3(tmp)
		return tmp
class NetPro(nn.Module):
	def __init__(self,hid_dim,code_size,owner_size):
		super(NetPro,self).__init__()
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
			nn.Linear(hid_dim,2),
			nn.LeakyReLU())
		self.code_layer1=nn.Sequential(
			nn.Linear(hid_dim,hid_dim),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.code_layer2=nn.Sequential(
			nn.Linear(hid_dim,2),
			nn.LeakyReLU())
		self.owner_layer1=nn.Sequential(
			nn.Linear(hid_dim,hid_dim),
			nn.ReLU(),
			nn.Dropout(0.5))
		self.owner_layer2=nn.Sequential(
			nn.Linear(hid_dim,2),
			nn.LeakyReLU())
		self.final=nn.Sequential(
			nn.Linear(6,2),
			nn.Sigmoid())
	def forward(self,codes,owners,others):
		code_emb=self.code_emb_layer(codes.to(torch.int64))
		owner_emb=self.owner_emb_layer(owners.to(torch.int64))
		mix=torch.cat([code_emb,owner_emb,others],dim=1)
		mix=self.mix_layer1(mix)
		mix=self.mix_layer2(mix)
		mix=self.mix_layer3(mix)

		code=self.code_layer1(code_emb)
		code=self.code_layer2(code)

		owner=self.owner_layer1(owner_emb)
		owner=self.owner_layer2(owner)

		tmp=torch.cat([mix,code,owner],dim=1)
		tmp=self.final(tmp)
		return tmp
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

class GraphNet(nn.Module):
	"""docstring for GraphNet"""
	def __init__(self,hid_dim,code_size,owner_size):
		super(GraphNet, self).__init__()
		
		# self.gc1=nn.Sequential(
		# 	GraphConvolution(owner_size,hid_dim),
		# 	nn.ReLU(),
		# 	nn.Dropout(0.5)
		# 	)
		self.gc1=GraphConvolution(owner_size,hid_dim)
		# self.gc2=GraphConvolution(hid_dim,hid_dim)


		self.code_emb_layer=nn.Embedding(code_size,hid_dim)
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
	def forward(self,codes,owners,others,owners_index,owners_adj):
		code_emb=self.code_emb_layer(codes.to(torch.int64))
		owner_gc=F.dropout(F.relu(self.gc1(owners,owners_adj)),0.5,training=self.training)
		# owner_gc=F.dropout(F.relu(self.gc2(owner_gc,owners_adj)),0.5,training=self.training)
		owner_emb=owner_gc[owners_index.to(torch.int64)]
		mix=torch.cat([code_emb,owner_emb,others],dim=1)
		mix=self.mix_layer1(mix)
		mix=self.mix_layer2(mix)
		tmp=torch.cat([mix,code_emb,owner_emb],dim=1)
		tmp=self.mix_layer3(tmp)
		return tmp
		