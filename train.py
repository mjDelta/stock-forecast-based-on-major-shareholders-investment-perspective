import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import optim
import math
import os
from models import *
import scipy.sparse as sp
from six.moves import cPickle as pickle
from utils import *
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")


data_path="E:/deqin/processed_data/HLD_Chgequity_success_second_market2_clean.csv"
out_dir="E:/deqin/saved_models190724/";mkdirs(out_dir)
load_model_path=os.path.join(out_dir,"epoch_200000.tar")
# load_model_path=""
hid_dim=50
# epochs=200001
epochs=0
batch_size=512
# train_rate=0.7
# val_rate=0.15
# test_rate=0.15
train_rate=0
val_rate=0
test_rate=1

train_xs,train_ys,val_xs,val_ys,test_xs,test_ys,codes_dict,owner_dict=load_data(data_path,train_rate,val_rate)


val_codes=torch.FloatTensor(val_xs[:,0]).to(device)
val_owners=torch.FloatTensor(val_xs[:,1]).to(device)
val_others=torch.FloatTensor(val_xs[:,2:]).to(device)
val_ys=torch.FloatTensor(val_ys).to(device)


model=NetPro2(50,len(codes_dict),len(owner_dict))
if load_model_path!="":
	model_sd=torch.load(load_model_path)
	model.load_state_dict(model_sd["model"])
	print("Load model from {}".format(load_model_path))
model=model.to(device)
model.train(mode=True)
optimizer=optim.SGD(model.parameters(),lr=1e-4,momentum=0.9,weight_decay=0.005)
criterion=nn.BCELoss()

iters=math.ceil(len(train_xs)/batch_size)
iter_losses=[]
train_epoch_losses=[]
val_epoch_losses=[]
val_epoch_accs=[]
for epoch in range(epochs):
	epoch_loss=0.
	model.train(mode=True)
	for it in range(iters):
		optimizer.zero_grad()
		start=it*batch_size
		end=(it+1)*batch_size
		batch_xs=train_xs[start:end]
		batch_ys=train_ys[start:end]

		# batch_xs=torch.FloatTensor(batch_xs).to(device)
		batch_codes=torch.FloatTensor(batch_xs[:,0]).to(device)
		batch_owners=torch.FloatTensor(batch_xs[:,1]).to(device)
		batch_others=torch.FloatTensor(batch_xs[:,2:]).to(device)
		batch_ys=torch.FloatTensor(batch_ys).to(device)

		pred_batch_ys=model(batch_codes,batch_owners,batch_others)

		batch_loss=criterion(pred_batch_ys,batch_ys)

		iter_losses.append(batch_loss.item())
		epoch_loss+=batch_loss.item()

		batch_loss.backward()
		optimizer.step()
	epoch_loss/=iters
	train_epoch_losses.append(epoch_loss)

	model.eval()
	val_pred_ys=model(val_codes,val_owners,val_others)
	val_loss=criterion(val_pred_ys,val_ys)
	val_epoch_losses.append(val_loss.item())
	val_acc=compute_acc(val_pred_ys.detach().cpu().numpy(),val_ys.detach().cpu().numpy())
	val_epoch_accs.append(val_acc)
	print("Epoch {} train loss:{} val loss:{} val acc:{}".format(epoch,round(epoch_loss,4),round(val_epoch_losses[-1],4),round(val_acc,4)))
	if epoch%5000==0:
		torch.save({
			"model":model.state_dict(),
			"optimizer":optimizer.state_dict()
			},os.path.join(out_dir,"epoch_{}.tar".format(epoch)))

test_codes=torch.FloatTensor(test_xs[:,0]).to(device)
test_owners=torch.FloatTensor(test_xs[:,1]).to(device)
test_others=torch.FloatTensor(test_xs[:,2:]).to(device)
test_ys=torch.FloatTensor(test_ys).to(device)

model.eval()
test_pred_ys=model(test_codes,test_owners,test_others)
test_acc=compute_acc(test_pred_ys.detach().cpu().numpy(),test_ys.detach().cpu().numpy())
print("Test accuracy: {}".format(test_acc))
df=pd.DataFrame()
df["iterations"]=np.arange(len(iter_losses))
df["loss"]=iter_losses
df.to_csv(os.path.join(out_dir,"iter_loss.csv"))

df=pd.DataFrame()
df["epochs"]=np.arange(len(train_epoch_losses))
df["loss"]=train_epoch_losses
df.to_csv(os.path.join(out_dir,"train_loss.csv"))

df=pd.DataFrame()
df["epochs"]=np.arange(len(val_epoch_losses))
df["loss"]=val_epoch_losses
df["acc"]=val_epoch_accs
df.to_csv(os.path.join(out_dir,"val_loss.csv"))