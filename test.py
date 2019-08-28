#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-25 22:44:00
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

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
from sklearn.manifold import TSNE
import json
import heapq

def getListMaxNumIndex(num_list,topk):

	min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))
	return min_num_index    

USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")


data_path="E:/deqin/processed_data/HLD_Chgequity_success_second_market2_clean.csv"
model_dir="E:/deqin/saved_models190724/"
load_model_path=os.path.join(model_dir,"epoch_200000.tar")

hid_dim=100
epochs=200001
batch_size=512
train_rate=0.7
val_rate=0.15
test_rate=0.15
topk=75
train_xs,train_ys,val_xs,val_ys,test_xs,test_ys,codes_dict,owner_dict_=load_data(data_path,train_rate,val_rate)
rng=np.random.RandomState(0)

# val_codes=torch.FloatTensor(val_xs[:,0]).to(device)
# val_owners=torch.FloatTensor(val_xs[:,1]).to(device)
# val_others=torch.FloatTensor(val_xs[:,2:]).to(device)
# val_ys=torch.FloatTensor(val_ys).to(device)


model=NetPro2(50,len(codes_dict),len(owner_dict_))
if load_model_path!="":
	model_sd=torch.load(load_model_path)
	model.load_state_dict(model_sd["model"])
	print("Load model from {}".format(load_model_path))
model=model.to(device)
model.eval()

# code_embeddings=model.code_emb_layer.weight.detach().cpu().numpy()
# code_decomposition=TSNE(n_components=2).fit_transform(code_embeddings)
owner_embeddings=model.owner_emb_layer.weight.detach().cpu().numpy()
owner_decomposition=TSNE(n_components=2).fit_transform(owner_embeddings)
owner_dict={}
for o,i in owner_dict_.items():

	owner_dict[o]=i
owner_labels=np.zeros(shape=(len(owner_dict),))##按个体投资者的机构投资者划分
for o,i in owner_dict.items():
	if len(o)>4:
		owner_labels[i]=1
colors=["blue","red","green"]

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,random_state=0).fit(owner_decomposition)
print(kmeans.cluster_centers_)

clusters={}
fig=plt.figure(figsize=(10,10),dpi=300)
for o,i in owner_dict.items():
	cluster_label=kmeans.labels_[i]
	if cluster_label not in clusters:
		clusters[cluster_label]=[]
	clusters[cluster_label].append(o)
	o_decom=owner_decomposition[i]

	x1=o_decom[0]
	x2=o_decom[1]
	# color=colors[int(owner_labels[i])]
	color=colors[int(cluster_label)]

	plt.scatter(x1,x2,c=color,alpha=0.7,s=4)
# plt.legend(["Blue: individual","Red: company"])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(model_dir,"owner_cluster.png"),bbox_inches = 'tight')

owner_clusters=pd.DataFrame()
owner_clusters["CLUSTER 0"]=clusters[0]
owner_clusters.to_csv(os.path.join(model_dir,"owners_clusters0.csv"),index=False)

owner_clusters=pd.DataFrame()
owner_clusters["CLUSTER 1"]=clusters[1]
owner_clusters.to_csv(os.path.join(model_dir,"owners_clusters1.csv"),index=False)

owner_clusters=pd.DataFrame()
owner_clusters["CLUSTER 2"]=clusters[2]
owner_clusters.to_csv(os.path.join(model_dir,"owners_clusters2.csv"),index=False)


reverse_owner_dict={}
for k,v in owner_dict.items():
	reverse_owner_dict[v]=k

def closest_topn(find_index,arr,topk):
	distances_sqaure=np.power(arr[:,0]-arr[find_index,0],2)+np.power(arr[:,1]-arr[find_index,1],2)
	distances=np.sqrt(distances_sqaure)
	topk_indexs=getListMaxNumIndex(list(distances),topk=topk)
	names=[]
	for idx in topk_indexs:
		name=reverse_owner_dict[idx]
		names.append(name)
	return names
out_dict={}
out_dict["nodes"]=[]
out_dict["links"]=[]
for key in range(3):
	cluster=clusters[key]
	for src in cluster:
		probs=rng.randint(1,11)
		
		node_dict={}
		node_dict["id"]=src
		node_dict["class"]="CLUSTER"+str(key)
		node_dict["group"]=key
		if len(src)>4:
			node_dict["size"]=4
		else:
			node_dict["size"]=2
		out_dict["nodes"].append(node_dict)
		for tg in cluster:
			if probs<9:continue
			if tg==src:
				continue
			link_dict={}
			link_dict["source"]=src
			link_dict["target"]=tg
			link_dict["value"]=key
			out_dict["links"].append(link_dict)
		src_index=owner_dict[src]
		topk_names=closest_topn(src_index,owner_decomposition,topk)
		for n in topk_names:
			if n in cluster:continue
			link_dict={}
			link_dict["source"]=src
			link_dict["target"]=n
			link_dict["value"]=key+3
			out_dict["links"].append(link_dict)			

with open(os.path.join(model_dir,"owners.json"),"w") as f:
	f.write(json.dumps(out_dict,ensure_ascii=False))



