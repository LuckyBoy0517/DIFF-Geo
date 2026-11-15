import json
import os
import random

random.seed(0)

from collections import defaultdict
fr1 = open('Osaka_landmark.txt','r',encoding='UTF-8')
output_str=".//output_data//"

landmark_info_dict={} # Get Info from landmark.txt

for line in fr1.readlines():
    ld_info = line.strip().split(sep=' ')
    ip = ld_info[0]
    lon = ld_info[1]
    lat = ld_info[2]
    city = ld_info[3]
    landmark_info_dict[ip]=[city,lon,lat]

#path = os.listdir(os.getcwd())

folder_city_name = 'hk_to_OS_trace'

file1 = open(output_str + 'Osaka_v4_ip_id.txt','w')
file2 = open(output_str +'Osaka_v4_edges.txt', 'w')
file3 = open(output_str + 'Osaka_v4_dstip_id_allinfo.txt', 'w')
fw = open(output_str + 'Osaka_v4_dst_path.txt','w')

allpath = os.getcwd()

ip_id_dict = defaultdict(int) # {ip:id}
ip_delay_dict = defaultdict(list) # list of delays which are from source to ip {id : [delay1,delay2,……]}
edge_dict = defaultdict(list) # {'id1,id2':[]}
source_delay_dict = defaultdict(list) # source to source delay , the value is always zero. {id : [0]}
dst_delay_dict = defaultdict(list) # source to source delay , the value is list {id : [delay1,delay2，……]}
edge_delay_dict = defaultdict(list) # an edge delay , is ip2's mindelay - ip1's mindelay. 
removed_dst_dict = defaultdict(list) # the delay which is outside normal limits that should be removed
srcip_id_dict = defaultdict(int) #{srcIP:srcID}
dstip_id_dict = defaultdict(int) #{dstIP:dstID}
dstip_routerlist = defaultdict(dict) # multi routerlist stored to the dstIP

tracepath = allpath + os.sep + folder_city_name
os.chdir(tracepath) # change the workplace
chpath = os.listdir(tracepath) #get the folder's data list
filei = 0
for chp in chpath:
    f1 = open(chp,"r")
    filei = filei + 1

    for line in f1:
        data = json.loads(line)
        if 'stop_reason' not in data.keys():
            continue
        if data['stop_reason'] != 'COMPLETED':
            continue

        dstIP = data['dst'] # get dst IP
        if dstIP in removed_dst_dict: # remove the IP that does not meet our requirements
            continue

        delay = float(data['hops'][-1]['rtt'])
        if delay > 7250:
            removed_dst_dict[dstIP] = 0
            continue

        srcIP = data['src']
        hops = data['hops'] #list
        full_hops = data['hop_count']
        alive_hops = len(hops)
        fail_detect_num = full_hops - alive_hops

        ip_id_dict.setdefault(srcIP , len(ip_id_dict)) # save srcIP into general list and get srcIP's ID
        srcip_id = ip_id_dict[srcIP]
        srcip_id_dict[srcIP] = srcip_id
        ip_delay_dict[srcIP] = [0]

        ip_id_dict.setdefault(dstIP , len(ip_id_dict)) # save dstIP into general list and get dstIP's ID
        dstip_id = ip_id_dict[dstIP]
        dstip_id_dict[dstIP] = dstip_id
        ip_delay_dict.setdefault(dstIP,[])
        ip_delay_dict[dstIP].append(delay)

        dstip_routerlist.setdefault(dstIP,{})
        dstip_routerlist[dstIP].setdefault(full_hops,[]) # {dstIP : {fullhops1:[[routerlist1],[routerlist2],……]},fullhops2:[[routerlist1],[routerlist2],……]}}
        router_list = [srcip_id]
        router_list.extend(["lost"] * full_hops)

        previous_id = srcip_id
        previous_rtt = 0
        limitflag = 0

        for hop in hops:
            hopIP = hop['addr']
            current_rtt = hop['rtt']
            probe_ttl = int(hop['probe_ttl']) #curIP's hop,ex. 2
            current_edge_delay = current_rtt - previous_rtt

            if(hopIP not in ip_id_dict):
                ip_id_dict[hopIP] = len(ip_id_dict)

            current_id = ip_id_dict[hopIP]
            temp_edge = str(previous_id)+','+str(current_id)
            ip_delay_dict.setdefault(hopIP,[])
            ip_delay_dict[hopIP].append(current_rtt)

            router_list[probe_ttl] = current_id
            edge_dict.setdefault(temp_edge,[])
            edge_dict[temp_edge].append(current_edge_delay)

            previous_id=current_id
            previous_rtt = current_rtt
        

        """
            Now , we have got current router_list , routes that fail to be detected are represented by 'lost'
            ex.
            routerlist = [srcip_id , 132 , 6 , 129 , lost , 441 , 506 , dstip_id]
        """

        flag = 0
        if len(dstip_routerlist[dstIP][full_hops]) == 0:
            dstip_routerlist[dstIP][full_hops].append(router_list)
            continue

        for router_list_id , old_router_list in enumerate(dstip_routerlist[dstIP][full_hops]):
            possible_router_list = old_router_list
            for k,c in enumerate(old_router_list):
                if router_list[k]!='lost':
                    if c=='lost':
                        possible_router_list[k] = router_list[k]
                    else:
                        if c!=router_list[k]:
                            flag = 1 # router_list is not the same path as old_router_list , so just break.
                            break

            if flag == 0:
                dstip_routerlist[dstIP][full_hops][router_list_id] = possible_router_list
                break
            else:
                if(router_list_id == len(dstip_routerlist[dstIP][full_hops])-1):
                    dstip_routerlist[dstIP][full_hops].append(router_list)
                    break
                else:
                    flag = 0 

# {dstIP : {fullhops1:[[routerlist1],[routerlist2],……]},fullhops2:[[routerlist1],[routerlist2],……]}}
"""
    dst_path.txt:
    dstIP,fullhops;lost_num;lostRouterloc;routerlist
    27.115.28.102,17;2;1,14,;0,lost,2,3,4,5,6,7,8,9,10,11,12,13,lost,14,15,1,
"""
for dstIP,content in dstip_routerlist.items():
    for full_hops,router_list_for_this_fullhops in content.items():
        for router_list in router_list_for_this_fullhops:
            lost_num = 0
            for router in router_list:
                router = str(router)
                if 'lost' in router:
                    lost_num += 1
            fw.write(dstIP + ',' + str(full_hops) + ';' + str(lost_num) + ';')
            for loc,temp_router in enumerate(router_list):
                if temp_router == 'lost':
                    fw.write(str(loc)+',')
            fw.write(';')
            for loc,temp_router in enumerate(router_list):
                fw.write(str(temp_router)+',')
            fw.write('\n')
fw.close()

"""
    file1: ip_id.txt        ip,id,mindelay
    file2: edges.txt        id1,id2,delay,ip1,ip2
    file3: dstip_id_allinfo.txt  dstip,id,city,lon/lat,lat/lon
"""
ip_id_mindelay_dict = {}
id_delay_dict = {}
for key,content in ip_delay_dict.items():
    ip = key
    id = ip_id_dict[ip]
    delay_list = content
    min_delay = min(delay_list)
    ip_id_mindelay_dict [id] = [ip,min_delay]
    id_delay_dict[id] = float(delay)
    file1.write(ip + ',' + str(id) + ',' + str(min_delay) + '\n')

full_edge_dict = {}
for key,content in edge_dict.items():
    id_id = key.split(',')
    id1 = int(id_id[0])
    id2 = int(id_id[1])
    ip1 = ip_id_mindelay_dict[id1][0]
    ip2 = ip_id_mindelay_dict[id2][0]
    delay1 = ip_id_mindelay_dict[id1][1]
    delay2 = ip_id_mindelay_dict[id2][1]
    nonnegative_numbers = [num for num in content if num > 0] # get nonnegative min delay ,if empty, then use below.
    if nonnegative_numbers:
        delay = min(nonnegative_numbers)
    else:
        delay = -1
    if delay < 0:
        delay = delay2 - delay1
    if delay < 0:
        delay = 0.001
    full_edge_dict[str(id1) + ',' +str(id2)] = delay
    file2.write(key + ',' + str(delay) +','+ip1+','+ip2+'\n')



for dstip, id_city in dstip_id_dict.items():
    id = id_city
    file3.write(dstip + ',' + str(id) + ',')
    if dstip in landmark_info_dict:
        content = landmark_info_dict[dstip]
    file3.write(content[0] + ',' + content[1] + ',' + content[2] + '\n')

file1.close()
file2.close()
file3.close()





