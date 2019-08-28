#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-25 13:16:33
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import optim
import math
import os
from models import *
from six.moves import cPickle as pickle
from utils import *
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")
def adjust_learning_rate(optimizer, lr):
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

data_path="E:/deqin/processed_data/HLD_Chgequity_success_second_market2_clean.csv"
out_dir="E:/deqin/saved_models190726_1gcnlayers/";mkdirs(out_dir)
data_dir="E:/deqin/saved_data/"
load_model_path=os.path.join(out_dir,"epoch_4000.tar")
# load_model_path=""
lr=1e-4
# lr=1e-2
hid_dim=100
epochs=100001
batch_size=1024
train_rate=0.7
val_rate=0.15
test_rate=0.15
train_xs,train_owner_indexs,train_ys,val_xs,val_owner_indexs,val_ys,test_xs,test_owner_indexs,test_ys,codes_dict,owner_dict,owner_adjacent,owners_onehot=load_data_for_graph(data_dir,train_rate,val_rate)

owner_adjacent=torch.FloatTensor(owner_adjacent).to(device)
owners_onehot=torch.FloatTensor(owners_onehot).to(device)
val_codes,val_owners,val_others=[],[],[]
for v_code,v_owner,v_other in zip(val_xs[:,0],val_xs[:,1],val_xs[:,2:]):
	val_codes.append(v_code)
	val_owners.append(v_owner)
	val_others.append(v_other)

code_dim=len(codes_dict)
owner_dim=len(owner_dict)
val_codes=torch.FloatTensor(val_codes).to(device)
val_owners=torch.FloatTensor(val_owners).to(device)
val_others=torch.FloatTensor(val_others).to(device)
val_owner_indexs=torch.FloatTensor(val_owner_indexs).to(device)

val_ys=torch.FloatTensor(val_ys).to(device)


model=GraphNet(50,code_dim,owner_dim)
if load_model_path!="":
	model_sd=torch.load(load_model_path)
	model.load_state_dict(model_sd["model"])
	print("Load model from {}".format(load_model_path))
model=model.to(device)
model.train(mode=True)
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
criterion=nn.BCELoss()

iters=math.ceil(len(train_xs)/batch_size)
iter_losses=[]
train_epoch_losses=[]
val_epoch_losses=[]
val_epoch_accs=[]
best_val_loss=9999
patience=500
nonupdate_cnt=0

for epoch in range(4001,epochs):
	if lr<1e-4:lr=1e-4
	epoch_loss=0.
	model.train(mode=True)
	for it in range(iters):
		optimizer.zero_grad()
		start=it*batch_size
		end=(it+1)*batch_size
		batch_xs=train_xs[start:end]
		batch_ys=train_ys[start:end]
		batch_owner_indexs=train_owner_indexs[start:end]

		batch_codes,batch_owners,batch_others=[],[],[]
		for b_code,b_owner,b_other in zip(batch_xs[:,0],batch_xs[:,1],batch_xs[:,2:]):
			batch_codes.append(b_code)
			batch_owners.append(b_owner)
			batch_others.append(b_other)

		batch_codes=torch.FloatTensor(batch_codes).to(device)
		batch_owners=torch.FloatTensor(batch_owners).to(device)
		batch_others=torch.FloatTensor(batch_others).to(device)
		batch_owner_indexs=torch.FloatTensor(batch_owner_indexs).to(device)
		batch_ys=torch.FloatTensor(batch_ys).to(device)

		pred_batch_ys=model(batch_codes,owners_onehot,batch_others,batch_owner_indexs,owner_adjacent)

		batch_loss=criterion(pred_batch_ys,batch_ys)

		iter_losses.append(batch_loss.item())
		epoch_loss+=batch_loss.item()

		batch_loss.backward()
		optimizer.step()
	epoch_loss/=iters
	train_epoch_losses.append(epoch_loss)

	model.eval()
	val_pred_ys=model(val_codes,owners_onehot,val_others,val_owner_indexs,owner_adjacent)
	val_loss=criterion(val_pred_ys,val_ys)
	val_epoch_losses.append(val_loss.item())
	val_acc=compute_acc(val_pred_ys.detach().cpu().numpy(),val_ys.detach().cpu().numpy())
	val_epoch_accs.append(val_acc)
	if epoch%100==0:
		print("Epoch {} train loss:{} val loss:{} val acc:{}".format(epoch,round(epoch_loss,4),round(val_epoch_losses[-1],4),round(val_acc,4)))
	if epoch%1000==0:
		torch.save({
			"model":model.state_dict(),
			"optimizer":optimizer.state_dict()
			},os.path.join(out_dir,"epoch_{}.tar".format(epoch)))
	if val_epoch_losses[-1]+0.0001<best_val_loss:
		print("[Best] Epoch {} train loss:{} val loss:{} val acc:{}".format(epoch,round(epoch_loss,4),round(val_epoch_losses[-1],4),round(val_acc,4)))
		best_val_loss=val_epoch_losses[-1]
		torch.save({
			"model":model.state_dict(),
			"optimizer":optimizer.state_dict()
			},os.path.join(out_dir,"best_model.tar".format(epoch)))	
		nonupdate_cnt=0	
	else:
		nonupdate_cnt+=1
	if nonupdate_cnt>=patience:
		lr=lr/2
		adjust_learning_rate(optimizer,lr)
		nonupdate_cnt=0
		print("Decrease lr to {}".format(lr))

test_codes,test_owners,test_others=[],[],[]
for t_code,t_owner,t_other in zip(test_xs[:,0],test_xs[:,1],test_xs[:,2:]):
	test_codes.append(t_code)
	test_owners.append(t_owner)
	test_others.append(t_other)

test_codes=torch.FloatTensor(test_codes).to(device)
test_owners=torch.FloatTensor(test_owners).to(device)
test_others=torch.FloatTensor(test_others).to(device)
test_owner_indexs=torch.FloatTensor(test_owner_indexs).to(device)
test_ys=torch.FloatTensor(test_ys).to(device)

model.eval()
test_pred_ys=model(test_codes,owners_onehot,test_others,test_owner_indexs,owner_adjacent)
test_acc=compute_acc(test_pred_ys.detach().cpu().numpy(),test_ys.detach().cpu().numpy())
print("Test accuracy: {}".format(test_acc))
df=pd.DataFrame()
df["iterations"]=np.arange(len(iter_losses))
df["loss"]=iter_losses
df.to_csv(os.path.join(out_dir,"iter_loss2.csv"))

df=pd.DataFrame()
df["epochs"]=np.arange(len(train_epoch_losses))
df["loss"]=train_epoch_losses
df.to_csv(os.path.join(out_dir,"train_loss2.csv"))

df=pd.DataFrame()
df["epochs"]=np.arange(len(val_epoch_losses))
df["loss"]=val_epoch_losses
df["acc"]=val_epoch_accs
df.to_csv(os.path.join(out_dir,"val_loss2.csv"))
