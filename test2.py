#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-27 22:57:40
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

hid_dim=50
epochs=200001
batch_size=512
train_rate=0.7
val_rate=0.15
test_rate=0.15
topk=25
train_xs,train_ys,val_xs,val_ys,test_xs,test_ys,codes_dict,owner_dict=load_data(data_path,train_rate,val_rate)
xs=np.vstack((train_xs,val_xs,test_xs))
ys=np.vstack((train_ys,val_ys,test_ys))
reverse_owner_dict={}
for k,v in owner_dict.items():
	reverse_owner_dict[v]=k

reverse_code_dict={}
for k,v in codes_dict.items():
	reverse_code_dict[v]=k
model=NetPro2(50,len(codes_dict),len(owner_dict))
if load_model_path!="":
	model_sd=torch.load(load_model_path)
	model.load_state_dict(model_sd["model"])
	print("Load model from {}".format(load_model_path))
model=model.to(device)
model.eval()

owner_embeddings=model.owner_emb_layer.weight.detach().cpu().numpy()
codes_embeddings=model.code_emb_layer.weight.detach().cpu().numpy()
joint_embeddings=[]
for x in xs:

	owner=owner_embeddings[int(x[1]),:]
	code=codes_embeddings[int(x[0]),:]
	joint=np.zeros(shape=(2*hid_dim,))
	joint[:hid_dim]=code
	joint[hid_dim:]=owner
	joint_embeddings.append(joint)
joint_embeddings=np.array(joint_embeddings)
joint_decomposition=TSNE(n_components=2).fit_transform(joint_embeddings)


from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=2,random_state=0).fit(joint_decomposition)
print(kmeans.cluster_centers_)
colors=["red","green"]
o_clusters={}
c_clusters={}
fig=plt.figure(figsize=(10,10),dpi=300)
for i,y in enumerate(ys):
	o=xs[i][1]
	c=xs[i][0]
	cluster_label=kmeans.labels_[i]
	if cluster_label not in o_clusters:
		o_clusters[cluster_label]=[]
		c_clusters[cluster_label]=[]
	o_clusters[cluster_label].append(reverse_owner_dict[o])
	c_clusters[cluster_label].append(reverse_code_dict[c])

	j_decom=joint_embeddings[i]

	x1=j_decom[0]
	x2=j_decom[1]
	color=colors[int(cluster_label)]

	plt.scatter(x1,x2,c=color,alpha=0.7,s=4)
# plt.legend(["Blue: individual","Red: company"])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(os.path.join(model_dir,"joint_cluster.png"),bbox_inches = 'tight')

owner_clusters=pd.DataFrame()
owner_clusters["C0_owners"]=o_clusters[0]
owner_clusters["C0_codes"]=c_clusters[0]
owner_clusters.to_csv(os.path.join(model_dir,"joint_clusters0.csv"),index=False)

owner_clusters=pd.DataFrame()
owner_clusters["C1_owners"]=o_clusters[1]
owner_clusters["C1_codes"]=c_clusters[1]
owner_clusters.to_csv(os.path.join(model_dir,"joint_clusters1.csv"),index=False)

# owner_clusters=pd.DataFrame()
# owner_clusters["CLUSTER 2"]=clusters[2]
# owner_clusters.to_csv(os.path.join(model_dir,"owners_clusters2_new.csv"),index=False)




# def closest_topn(find_index,arr,topk):
# 	distances_sqaure=np.power(arr[:,0]-arr[find_index,0],2)+np.power(arr[:,1]-arr[find_index,1],2)
# 	distances=np.sqrt(distances_sqaure)
# 	topk_indexs=getListMaxNumIndex(list(distances),topk=topk)
# 	names=[]
# 	for idx in topk_indexs:
# 		name=reverse_owner_dict[idx]
# 		names.append(name)
# 	return names
# out_dict={}
# out_dict["nodes"]=[]
# out_dict["links"]=[]
# for key in range(3):
# 	cluster=clusters[key]
# 	for src in cluster:
# 		node_dict={}
# 		node_dict["id"]=src
# 		node_dict["class"]="CLUSTER"+str(key)
# 		node_dict["group"]=key
# 		node_dict["size"]=5
# 		out_dict["nodes"].append(node_dict)
# 		for tg in cluster:
# 			if tg==src:
# 				continue
# 			link_dict={}
# 			link_dict["source"]=src
# 			link_dict["target"]=tg
# 			link_dict["value"]=key
# 			out_dict["links"].append(link_dict)
# 		src_index=owner_dict[src]
# 		topk_names=closest_topn(src_index,owner_decomposition,topk)
# 		for n in topk_names:
# 			if n in cluster:continue
# 			link_dict={}
# 			link_dict["source"]=src
# 			link_dict["target"]=n
# 			link_dict["value"]=key+3
# 			out_dict["links"].append(link_dict)			

# with open(os.path.join(model_dir,"data.json"),"w") as f:
# 	f.write(json.dumps(out_dict,ensure_ascii=False))



