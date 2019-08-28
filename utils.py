#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-25 13:13:14
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
import numpy as np
import pandas as pd
import os
from six.moves import cPickle as pickle
import scipy.sparse as sp
def save_dict(di_,filename):
	with open(filename,"wb") as f:
		pickle.dump(di_,f)
def load_dict(filename):
	with open(filename,"rb") as f:
		ret_di=pickle.load(f)
	return ret_di
def mkdirs(dirs):
	if not os.path.exists(dirs):
		os.makedirs(dirs)
def to_onehot(x,max=2):
	onehot=np.zeros(shape=(max,))
	onehot[x]=1
	return onehot
def load_data(data_path,train_rate,val_rate):
	df_org=pd.read_csv(data_path,header=0,engine="python")
	df=df_org.dropna()
	codes=sorted(list(set(df["code"].values)))
	owners=sorted(list(set(df["S0802a"].values)))

	codes_dict={code:i for i,code in enumerate(codes)}
	owner_dict={owner:i for i,owner in enumerate(owners)}
	xs=[]
	ys=[]
	for i in range(len(df)):
		raw_x=df.iloc[i].values
		x=[codes_dict[raw_x[0]],owner_dict[raw_x[2]],float(raw_x[4]),int(raw_x[7])-1]
		y=to_onehot(raw_x[-1])
		xs.append(x)
		ys.append(y)
	xs=np.array(xs)
	ys=np.array(ys)

	rng=np.random.RandomState(0)
	all_idxs=np.arange(len(xs))
	rng.shuffle(all_idxs)
	train_idxs=all_idxs[:int(len(xs)*train_rate)]
	val_idxs=all_idxs[int(len(xs)*train_rate):int(len(xs)*(train_rate+val_rate))]
	test_idxs=all_idxs[int(len(xs)*(train_rate+val_rate)):]

	train_xs=xs[train_idxs];train_ys=ys[train_idxs]
	val_xs=xs[val_idxs];val_ys=ys[val_idxs]
	test_xs=xs[test_idxs];test_ys=ys[test_idxs]
	return train_xs,train_ys,val_xs,val_ys,test_xs,test_ys,codes_dict,owner_dict
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def saved_data_for_graph(data_path,out_dir):
	df_org=pd.read_csv(data_path,header=0,engine="python")
	df=df_org.dropna()
	codes=sorted(list(set(df["code"].values)))
	owners=sorted(list(set(df["S0802a"].values)))

	codes_dict={code:i for i,code in enumerate(codes)}
	owner_dict={owner:i for i,owner in enumerate(owners)}

	owner_adjacent=np.zeros(shape=(len(owners),len(owners)))

	for i,owner in enumerate(owners):
		print("{}/{}".format(i,len(owners)))
		owner_df=df[df["S0802a"]==owner]
		owned_codes=list(set(owner_df["code"].values))
		for j,neighbor in enumerate(owners):
			if neighbor==owner:
				continue
			neighbor_df=df[df["S0802a"]==neighbor]
			neighbor_codes=list(set(neighbor_df["code"].values))
			cnt=0
			for o_code in owned_codes:
				if o_code in neighbor_codes:
					cnt+=1
			weight=cnt/len(owned_codes)
			owner_adjacent[i,j]=weight
			if weight!=0:
				print(weight,owned_codes,neighbor_codes)
			# break
		# break
	owner_adjacent=normalize(owner_adjacent+sp.eye(owner_adjacent.shape[0]))
	xs=[]
	ys=[]
	onwer_indexs=[]
	for i in range(len(df)):
		raw_x=df.iloc[i].values
		x=[codes_dict[raw_x[0]],to_onehot(owner_dict[raw_x[2]],max=len(owners)),float(raw_x[4]),int(raw_x[7])-1]
		y=to_onehot(raw_x[-1])
		xs.append(x)
		ys.append(y)
		onwer_indexs.append(owner_dict[raw_x[2]])
	onwer_indexs=np.array(onwer_indexs)
	xs=np.array(xs)
	ys=np.array(ys)
	print(xs.sum())
	print(ys.sum())
	print(codes_dict["300339.SZ"])
	print(owner_dict["陈强"])
	print(owner_adjacent.sum())
	print(onwer_indexs[100])

	np.save(os.path.join(out_dir,"xs.npy"),xs)
	np.save(os.path.join(out_dir,"ys.npy"),ys)
	np.save(os.path.join(out_dir,"onwer_indexs.npy"),onwer_indexs)

	save_dict(codes_dict,os.path.join(out_dir,"codes_dict.pkl"))
	save_dict(owner_dict,os.path.join(out_dir,"owner_dict.pkl"))

	np.save(os.path.join(out_dir,"owner_adjacent.npy"),owner_adjacent)

def load_data_for_graph(data_dir,train_rate,val_rate):
	xs=np.load(os.path.join(data_dir,"xs.npy"),allow_pickle=True)
	ys=np.load(os.path.join(data_dir,"ys.npy"))
	onwer_indexs=np.load(os.path.join(data_dir,"onwer_indexs.npy"))


	codes_dict=load_dict(os.path.join(data_dir,"codes_dict.pkl"))
	owner_dict=load_dict(os.path.join(data_dir,"owner_dict.pkl"))

	owner_adjacent=np.load(os.path.join(data_dir,"owner_adjacent.npy"))	
	owners_onehot=np.identity(len(owner_dict))

	# print(xs.sum())
	# print(ys.sum())
	# print(codes_dict["300339.SZ"])
	# print(owner_dict["陈强"])
	# print(owner_adjacent.sum())
	rng=np.random.RandomState(0)
	all_idxs=np.arange(len(xs))
	rng.shuffle(all_idxs)
	train_idxs=all_idxs[:int(len(xs)*train_rate)]
	val_idxs=all_idxs[int(len(xs)*train_rate):int(len(xs)*(train_rate+val_rate))]
	test_idxs=all_idxs[int(len(xs)*(train_rate+val_rate)):]

	train_xs=xs[train_idxs];train_ys=ys[train_idxs];train_owner_indexs=onwer_indexs[train_idxs]
	val_xs=xs[val_idxs];val_ys=ys[val_idxs];val_owner_indexs=onwer_indexs[val_idxs]
	test_xs=xs[test_idxs];test_ys=ys[test_idxs];test_owner_indexs=onwer_indexs[test_idxs]
	return train_xs,train_owner_indexs,train_ys,val_xs,val_owner_indexs,val_ys,test_xs,test_owner_indexs,test_ys,codes_dict,owner_dict,owner_adjacent,owners_onehot


def compute_acc(pred,true):
	preds=np.argmax(pred,axis=1)
	trues=np.argmax(true,axis=1)
	cnter=0
	for p,t in zip(preds,trues):
		if p==t:
			cnter+=1
	return cnter/len(pred)
if __name__=="__main__":
	data_path="E:/deqin/processed_data/HLD_Chgequity_success_second_market2_clean.csv"
	out_dir="E:/deqin/saved_data/";mkdirs(out_dir)
	saved_data_for_graph(data_path,out_dir)
