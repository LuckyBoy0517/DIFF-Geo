import json
import os
import numpy as np
import networkx as nx
import pandas as pd
import random

from collections import defaultdict

random.seed(0)

"""
    dst_path.txt:
    dstIP,fullhops;lost_num;lostRouterloc;routerlist
    27.115.28.102,17;2;1,14,;0,lost,2,3,4,5,6,7,8,9,10,11,12,13,lost,14,15,1,
    ip_id.txt        ip,id,mindelay
    edges.txt        id1,id2,delay,ip1,ip2
"""

fr1 = open('output_data/Osaka_v4_dst_path.txt', 'r', encoding='UTF-8')
fr2 = open('output_data/Osaka_v4_edges.txt', 'r', encoding='UTF-8')
fr3 = open('output_data/Osaka_v4_ip_id.txt', 'r', encoding='UTF-8')
fw = open('output_data/Osaka_v4_edges_remove_lost.txt', 'w',encoding='utf-8')

id_delay_dict = {}
for line in fr3.readlines():#get delay from source to ip
    str_list = line.strip('\n').split(',')
    id = str_list[1]
    delay = str_list[2]
    ip = str_list[0]
    id_delay_dict[id]=float(delay)

full_edge_dict = {}
for line in fr2.readlines():#get delay from an edge.
    str_list = line.strip('\n').split(',')
    id1 = str_list[0]
    id2 = str_list[1]
    delay = str_list[2]
    ip1 = str_list[3]
    ip2 = str_list[4]
    full_edge_dict[id1+','+id2]=delay


"""
    edges_remove_lost.txt:
    Merge edges that contain "lost"

"""
processed_edge_dict = {}
processed_edge_dict2 = {}
for line in fr1.readlines():
    str_list = line.strip("\n").split(";")
    dstip_fullhops = str_list[0].split(',') # [dstip,fullhops]
    dstip = dstip_fullhops[0]

    lost_num = str_list[1]
    lost_list = str_list[2]
    router_list = str_list[3].split(',')[:-1]

    lost_count = 0 # count how many 'lost' an edge contains
    pre_router = router_list[0]

    for i, router in enumerate(router_list[1: ]):
        temp_router = router
        if(temp_router == 'lost'):
            lost_count += 1
            continue
        temp_edge = pre_router + ',' + temp_router

        if temp_edge in processed_edge_dict2:
            lost_count = 0
            pre_router = temp_router
            continue

        if temp_edge not in processed_edge_dict:
            if temp_edge in full_edge_dict:
                delay = full_edge_dict[temp_edge]
                if float(delay) > 7200:
                    continue
            else:
                delay1 = id_delay_dict[pre_router]
                delay2 = id_delay_dict[temp_router]
                delay = float(delay2) - float(delay1)
                if delay > 7200:
                    continue
            processed_edge_dict[temp_edge] = [delay,lost_count]
        
        else:
            if lost_count < processed_edge_dict[temp_edge][1]:
                processed_edge_dict[temp_edge][1] = lost_count
            if lost_count == 0: #if lostcount is zero , we should add this edge into edge set2.
                processed_edge_dict2[temp_edge] = [processed_edge_dict[temp_edge][0],0]
        
        lost_count = 0
        pre_router = temp_router

for k,v in processed_edge_dict.items():
    temp_edge = k
    [pre_router,temp_router] = temp_edge.split(',')

    delay = v[0]
    lost_count = v[1]
    fw.write(str(pre_router)+','+str(temp_router)+','+str(lost_count)+','+str(delay)+'\n')

fr1.close()
fr2.close()
fr3.close()
fw.close()