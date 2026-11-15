import json
import os
import numpy as np
import networkx as nx
import pandas as pd
import random

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

"""
    edges_remove_lost.txt       id1,id2,lost_count,delay
    ip_id.txt        ip,id,mindelay
    ip_feature.txt     ip,id,norm_delay,norm_ip_1,norm_ip_2,norm_ip_3,norm_ip_4
    edge_feature.txt      id1,id2,norm_delay,norm_lost,norm_ip0_1 ...,norm_ip1_1,...
"""
fr1 = open('output_data/Osaka_v4_edges_remove_lost.txt','r',encoding='UTF-8')
fr2 = open('output_data/Osaka_v4_ip_id.txt','r',encoding='UTF-8')
fw1 = open('output_data/Osaka_v4_ip_feature.txt','w',encoding='utf-8')
fw2 = open('output_data/Osaka_v4_edge_feature.txt','w',encoding='utf-8')

node_list = []
id_ip_dict = defaultdict(str)
for line in fr2.readlines():
    str_list = line.strip().split(sep=',')
    ip = str_list[0]

    ip_list = ip.split('.')

    id = int(str_list[1])
    id_ip_dict[id] = ip
    delay = float(str_list[2])
    node_list.append([ip,id,delay,float(ip_list[0]),float(ip_list[1]),float(ip_list[2]),float(ip_list[3])])

# use Minmax normalize method
node_list = np.array(node_list,dtype = object)
temp_normal = (node_list[:, 2:7] - node_list[:, 2:7].min(axis=0)) / (node_list[:, 2:7].max(axis=0) - node_list[:, 2:7].min(axis=0))  
node_feature = np.column_stack((node_list[: ,0:2],temp_normal)).tolist()
for node_feat in node_feature:
    ip = str(node_feat[0])
    id = str(node_feat[1])
    norm_delay = str(node_feat[2])
    norm_ip_1 = str(node_feat[3])
    norm_ip_2 = str(node_feat[4])
    norm_ip_3 = str(node_feat[5])
    norm_ip_4 = str(node_feat[6])
    fw1.write(ip+','+id+','+norm_delay+','+norm_ip_1+','+norm_ip_2+','+norm_ip_3+','+norm_ip_4+'\n')

edge_list = []

count = 0
for line in fr1.readlines():
    str_list = line.strip().split(sep=",")
    id1 = int(str_list[0])
    id2 = int(str_list[1])
    ip1 = id_ip_dict[id1]
    ip2 = id_ip_dict[id2]
    lost_count = float(str_list[2])
    ip1_list = ip1.split('.')
    ip2_list = ip2.split('.')
    delay = float(str_list[3])
    edge_list.append([id1,id2,delay,lost_count,float(ip1_list[0]),float(ip1_list[1]),float(ip1_list[2]),
                      float(ip1_list[3]),float(ip2_list[0]),float(ip2_list[1]),float(ip2_list[2]),float(ip2_list[3])])

edge_list = np.array(edge_list , dtype=object)
temp_normal = (edge_list[:, 2:12] - edge_list[:, 2:12].min(axis=0)) / (edge_list[:, 2:12].max(axis=0) - edge_list[:, 2:12].min(axis=0))  
edge_feature = np.column_stack((edge_list[: ,0:2],temp_normal)).tolist()

for edge_feat in edge_feature:
    id1 = str(edge_feat[0])
    id2 = str(edge_feat[1])
    norm_delay = str(edge_feat[2])
    norm_lost_count = str(edge_feat[3])
    norm_ip1_1 = str(edge_feat[4])
    norm_ip1_2 = str(edge_feat[5])
    norm_ip1_3 = str(edge_feat[6])
    norm_ip1_4 = str(edge_feat[7])
    norm_ip2_1 = str(edge_feat[8])
    norm_ip2_2 = str(edge_feat[9])
    norm_ip2_3 = str(edge_feat[10])
    norm_ip2_4 = str(edge_feat[11])
    fw2.write(id1+','+id2+','+norm_delay+','+norm_lost_count+','+norm_ip1_1+','+norm_ip1_2+
              ','+norm_ip1_3+','+norm_ip1_4+','+norm_ip2_1+','+norm_ip2_2+
              ','+norm_ip2_3+','+norm_ip2_4+'\n')
fw1.close()
fw2.close()