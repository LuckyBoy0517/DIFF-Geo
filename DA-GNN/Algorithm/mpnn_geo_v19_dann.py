# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import dgl
import random as rd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import itertools
import argparse
from math import radians, cos, sin, asin, sqrt
from sklearn import preprocessing


def get_min_err(err_list, err_list1, err_list2):
    min_err = min(err_list)
    ind = err_list.index(min_err)
    min_err1 = err_list1[ind]
    min_err2 = err_list2[ind]
    return min_err, min_err1, min_err2, ind


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    dgl.random.seed(seed)


setup_seed(0)


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def get_node_attr(filepath):
    fr1 = open(filepath, 'r')
    node_attr_list = []
    for line in fr1.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        temp_node_attr = list(map(eval, str_list[2:7]))
        node_attr_list.append(temp_node_attr)
    node_attr_array = np.array(node_attr_list)
    return node_attr_array


def build_graph(filepath):
    fr1 = open(filepath, 'r')
    src_list = []
    dst_list = []
    edge_attr_list = []
    for line in fr1.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        temp_src = int(str_list[0])
        temp_dst = int(str_list[1])
        temp_node_attr = list(map(eval, str_list[2:12]))
        src_list.append(temp_src)
        dst_list.append(temp_dst)
        edge_attr_list.append(temp_node_attr)
        src_list.append(temp_dst)
        dst_list.append(temp_src)
        edge_attr_list.append(temp_node_attr)

    src = np.array(src_list)
    dst = np.array(dst_list)
    edge_attr_array = np.array(edge_attr_list)
    return dgl.DGLGraph((src, dst)), edge_attr_array


def load_reachability_from_traceroute(traceroute_path, ip_feature_path):
    dst_reachable = {}
    with open(traceroute_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if 'stop_reason' in d and 'dst' in d:
                    reachable = 1 if d['stop_reason'] == 'COMPLETED' else 0
                    dst_reachable[d['dst']] = reachable
            except:
                pass

    ip_to_nodeid = {}
    with open(ip_feature_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            ip = parts[0]
            nodeid = int(parts[1])
            ip_to_nodeid[ip] = nodeid

    nodeid_reachable = {}
    for ip, reachable in dst_reachable.items():
        if ip in ip_to_nodeid:
            nodeid_reachable[ip_to_nodeid[ip]] = reachable

    return nodeid_reachable


def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    if kernel == 'rbf':
        C = 2 * sigma ** 2
        K_xx = torch.exp(-torch.clamp((rx.t() + rx - 2 * xx) / C, min=0))
        K_yy = torch.exp(-torch.clamp((ry.t() + ry - 2 * yy) / C, min=0))
        K_xy = torch.exp(-torch.clamp((rx.t() + ry - 2 * xy) / C, min=0))
    else:  # linear
        K_xx = xx
        K_yy = yy
        K_xy = xy

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


target_scaler1 = preprocessing.MinMaxScaler()
target_scaler2 = preprocessing.MinMaxScaler()
target_lat_label_dict = {}
target_lon_label_dict = {}
new_target_lat_label_dict = {}
new_target_lon_label_dict = {}
train_nodeid_list = []
val_nodeid_list = []
test_nodeid_list = []


def train_test_val(filepath_train, filepath_val, filepath_test):

    fr3 = open(filepath_train, 'r', encoding='UTF-8')
    fr4 = open(filepath_val, 'r', encoding='UTF-8')
    fr5 = open(filepath_test, 'r', encoding='UTF-8')

    target_lat_label_dict.clear()
    target_lon_label_dict.clear()

    train_lat_array = []
    train_lon_array = []
    for line in fr3.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        lon_label = float(str_list[3])
        lat_label = float(str_list[4])
        train_lat_array.append(lat_label)
        train_lon_array.append(lon_label)
        train_nodeid_list.append(node_id)
        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label

    for line in fr4.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        lon_label = float(str_list[3])
        lat_label = float(str_list[4])
        val_nodeid_list.append(node_id)
        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label

    for line in fr5.readlines():
        str_list = line.strip('\r\n').split(sep=',')
        node_id = int(str_list[1])
        lon_label = float(str_list[3])
        lat_label = float(str_list[4])
        test_nodeid_list.append(node_id)
        target_lat_label_dict[node_id] = lat_label
        target_lon_label_dict[node_id] = lon_label

    target_scaler1.fit(np.array(train_lat_array).reshape(-1, 1))
    target_scaler2.fit(np.array(train_lon_array).reshape(-1, 1))

    all_lat = [target_lat_label_dict[k] for k in target_lat_label_dict]
    all_lon = [target_lon_label_dict[k] for k in target_lon_label_dict]

    new_label_lat_array = target_scaler1.transform(np.array(all_lat).reshape(-1, 1)).astype(np.float32)
    new_label_lon_array = target_scaler2.transform(np.array(all_lon).reshape(-1, 1)).astype(np.float32)

    i = 0
    for k in target_lat_label_dict:
        new_target_lat_label_dict[k] = new_label_lat_array[i]
        new_target_lon_label_dict[k] = new_label_lon_array[i]
        i += 1

    lat_train_list, lon_train_list = [], []
    new_lat_train_list, new_lon_train_list = [], []
    lat_val_list, lon_val_list = [], []
    new_lat_val_list, new_lon_val_list = [], []
    lat_test_list, lon_test_list = [], []
    new_lat_test_list, new_lon_test_list = [], []

    for node_id in train_nodeid_list:
        lat_train_list.append(target_lat_label_dict[node_id])
        lon_train_list.append(target_lon_label_dict[node_id])
        new_lat_train_list.append(new_target_lat_label_dict[node_id])
        new_lon_train_list.append(new_target_lon_label_dict[node_id])

    for node_id in val_nodeid_list:
        lat_val_list.append(target_lat_label_dict[node_id])
        lon_val_list.append(target_lon_label_dict[node_id])
        new_lat_val_list.append(new_target_lat_label_dict[node_id])
        new_lon_val_list.append(new_target_lon_label_dict[node_id])

    for node_id in test_nodeid_list:
        lat_test_list.append(target_lat_label_dict[node_id])
        lon_test_list.append(target_lon_label_dict[node_id])
        new_lat_test_list.append(new_target_lat_label_dict[node_id])
        new_lon_test_list.append(new_target_lon_label_dict[node_id])

    train_labeled_nodes = torch.tensor(train_nodeid_list)
    lat_train_labels = torch.tensor(lat_train_list)
    lon_train_labels = torch.tensor(lon_train_list)
    new_lat_train_labels = torch.tensor(new_lat_train_list)
    new_lon_train_labels = torch.tensor(new_lon_train_list)

    val_labeled_nodes = torch.tensor(val_nodeid_list)
    lat_val_labels = torch.tensor(lat_val_list)
    lon_val_labels = torch.tensor(lon_val_list)
    new_lat_val_labels = torch.tensor(new_lat_val_list)
    new_lon_val_labels = torch.tensor(new_lon_val_list)

    test_labeled_nodes = torch.tensor(test_nodeid_list)
    lat_test_labels = torch.tensor(lat_test_list)
    lon_test_labels = torch.tensor(lon_test_list)
    new_lat_test_labels = torch.tensor(new_lat_test_list)
    new_lon_test_labels = torch.tensor(new_lon_test_list)

    return [train_labeled_nodes, new_lat_train_labels, new_lon_train_labels,
            val_labeled_nodes, new_lat_val_labels, new_lon_val_labels,
            test_labeled_nodes, new_lat_test_labels, new_lon_test_labels,
            lat_train_labels, lon_train_labels,
            lat_val_labels, lon_val_labels,
            lat_test_labels, lon_test_labels]


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=3):
    assert expected_order in ['acc', 'dec']
    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1
    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, best_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DANN-MPNN geo localization.")
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--node_embedding_size', type=int, default=128)
    parser.add_argument('--node_hidden_size', type=int, default=128)
    parser.add_argument('--edge_hidden_size', type=int, default=4)
    parser.add_argument('--fc_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_flag', type=int, default=1)
    parser.add_argument('--num_step_message_passing', type=int, default=3)
    parser.add_argument('--regs', type=float, default=0.0005)
    parser.add_argument('--gnn_dropout', type=float, default=0.3)
    parser.add_argument('--edge_dropout', type=float, default=0.2)
    parser.add_argument('--nn_dropout1', type=float, default=0.3)
    parser.add_argument('--nn_dropout2', type=float, default=0.0)
    parser.add_argument('--aggregator_type', type=str, default='sum')
    parser.add_argument('--loss1_weight', type=float, default=1)

    parser.add_argument('--use_dann', type=int, default=1, help='Whether to use domain adversarial training (0/1)')
    parser.add_argument('--domain_weight', type=float, default=0.1, help='Domain classification loss weight')
    parser.add_argument('--domain_hidden_dim', type=int, default=64)
    parser.add_argument('--domain_dp', type=float, default=0.3)

    parser.add_argument('--use_mmd', type=int, default=1, help='Whether to use MMD regularization (0/1)')
    parser.add_argument('--mmd_weight', type=float, default=0.01, help='MMD loss weight')
    # Label Smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.01, help='Label smoothing magnitude')

    parser.add_argument('--use_reachability_feat', type=int, default=1, help='Whether to use reachability feature (0/1)')

    args = parser.parse_args()
    use_dann = bool(args.use_dann)
    use_mmd = bool(args.use_mmd)
    use_reachability_feat = bool(args.use_reachability_feat)

    aggregator_type = args.aggregator_type
    node_embedding_size = args.node_embedding_size
    node_hidden_size = args.node_hidden_size
    edge_hidden_size = args.edge_hidden_size
    fc_size = args.fc_size
    regs = args.regs
    gnn_dropout = args.gnn_dropout
    edge_dropout = args.edge_dropout
    nn_dropout1 = args.nn_dropout1
    nn_dropout2 = args.nn_dropout2
    loss1_weight = args.loss1_weight
    gpu_id = args.gpuid
    epochs = args.epochs
    lr = args.lr
    print_flag = args.print_flag
    num_step_message_passing = args.num_step_message_passing

    use_cuda = args.gpuid >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpuid)
    print(use_cuda)

    filepath = os.path.split(os.path.realpath(__file__))[0]
    from time import time

    stamp = int(time())
    output_dir = filepath + '/output_train70_dann/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_path1 = output_dir + str(stamp) + '_loss.txt'

    fw1 = open(result_path1, 'w')
    perf_str = ('aggregator_type=%s,gnn_dropout=%s,edge_dropout=%s,nn_dropout1=%s,nn_dropout2=%s,'
                'regs=%s,node_embedding_size=%s,node_hidden_size=%s,edge_hidden_size=%s,fc_size=%s,'
                'num_step_message_passing=%s,loss1_weight=%s,lr=%.4f,'
                'use_dann=%s,domain_weight=%s,use_mmd=%s,mmd_weight=%s,'
                'label_smoothing=%s,use_reachability_feat=%s,domain_dp=%s\n') % (
        aggregator_type, gnn_dropout, edge_dropout, nn_dropout1, nn_dropout2,
        regs, node_embedding_size, node_hidden_size, edge_hidden_size, fc_size,
        num_step_message_passing, loss1_weight, lr,
        use_dann, args.domain_weight, use_mmd, args.mmd_weight,
        args.label_smoothing, use_reachability_feat,args.domain_dp)

    fw1.write(perf_str)
    if print_flag > 0:
        print(perf_str)
    fw1.flush()

    med_err_res, ave_err_res, rmse_err_res = [], [], []
    med_err_res_eval, ave_err_res_eval, rmse_err_res_eval = [], [], []
    loss_loger, error_loger = [], []
    stopping_step = 0

    G, edge_attr_array = build_graph(filepath + '/data_0201/Osaka_v4_edge_feature.txt')
    node_attr_array = get_node_attr(filepath + '/data_0201/Osaka_v4_ip_feature.txt')
    num_nodes = G.number_of_nodes()

    if print_flag > 0:
        print('We have %d nodes.' % num_nodes)
        print('We have %d edges.' % G.number_of_edges())

    traceroute_path = filepath + '/../Osaka.txt.txt'
    ip_feature_path = filepath + '/data_0201/Osaka_v4_ip_feature.txt'
    nodeid_reachable = load_reachability_from_traceroute(traceroute_path, ip_feature_path)

    if print_flag > 0:
        r_count = sum(1 for v in nodeid_reachable.values() if v == 1)
        u_count = sum(1 for v in nodeid_reachable.values() if v == 0)
        print(f'Reachability map: reachable={r_count}, unreachable={u_count}, total_targets={len(nodeid_reachable)}')


    if use_reachability_feat:
        reachability_feat = np.full((num_nodes, 1), 0.5, dtype=np.float32)
        for node_id, reachable in nodeid_reachable.items():
            if node_id < num_nodes:
                reachability_feat[node_id, 0] = float(reachable)
        if print_flag > 0:
            r_f = sum(1 for v in reachability_feat if v[0] > 0.5)
            u_f = sum(1 for v in reachability_feat if v[0] < 0.5)
            print(f'Reachability feature: reachable={r_f}, unreachable={u_f}, intermediate={num_nodes - r_f - u_f}')
    else:
        reachability_feat = np.zeros((num_nodes, 1), dtype=np.float32)


    all_target_nodeids = list(nodeid_reachable.keys())
    all_target_domain_labels = torch.tensor([float(nodeid_reachable[nid]) for nid in all_target_nodeids],
                                             dtype=torch.float32)
    all_target_nodeids_tensor = torch.tensor(all_target_nodeids)

    embed = nn.Embedding(num_nodes, node_embedding_size)
    G.ndata['feat'] = embed.weight


    extra_feat_dim = 1 if use_reachability_feat else 0
    node_in_feats = node_embedding_size + 5 + extra_feat_dim

    from mpnn_v9 import MPNN
    net = MPNN(aggregator_type, node_in_feats, node_hidden_size, 10, edge_hidden_size,
               num_step_message_passing, gnn_dropout, edge_dropout, nn_dropout1,
               use_dann=use_dann, domain_hidden_dim=args.domain_hidden_dim,
               domain_dp=args.domain_dp)

    if use_cuda:
        net.cuda()

    data_prefix = filepath + '/data_0201/Osaka_v4_dstip_id_allinfo'
    train_file = data_prefix + '_train.txt'
    val_file = data_prefix + '_val.txt'
    test_file = data_prefix + '_test.txt'

    (train_labeled_nodes, lat_train_labels, lon_train_labels,
     val_labeled_nodes, lat_val_labels, lon_val_labels,
     test_labeled_nodes, lat_test_labels, lon_test_labels,
     old_lat_train_labels, old_lon_train_labels,
     old_lat_val_labels, old_lon_val_labels,
     old_lat_test_labels, old_lon_test_labels) = train_test_val(train_file, val_file, test_file)


    test_reachable_list = []
    for nid in test_nodeid_list:
        test_reachable_list.append(nodeid_reachable.get(nid, 0))
    val_reachable_list = [nodeid_reachable.get(nid, 1) for nid in val_nodeid_list]

    if print_flag > 0:
        train_r = sum(1 for nid in train_nodeid_list if nodeid_reachable.get(nid, 0) == 1)
        train_u = len(train_nodeid_list) - train_r
        test_r = sum(test_reachable_list)
        test_u = len(test_reachable_list) - test_r
        print(f'Train: {len(train_nodeid_list)} (reachable={train_r}, unreachable={train_u})')
        print(f'Val: {len(val_nodeid_list)} (all reachable)')
        print(f'Test: {len(test_nodeid_list)} (reachable={test_r}, unreachable={test_u})')

    optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=lr,
                                 weight_decay=regs)
    loss_fn = nn.MSELoss()
    domain_loss_fn = nn.BCELoss()
    if use_cuda:
        loss_fn = loss_fn.cuda()
        domain_loss_fn = domain_loss_fn.cuda()


    node_attr_tensor = torch.from_numpy(node_attr_array).to(torch.float32)
    reachability_tensor = torch.from_numpy(reachability_feat).to(torch.float32)

    inputs = embed.weight
    if use_reachability_feat:
        node_feat = torch.cat([inputs, node_attr_tensor, reachability_tensor], dim=1)
    else:
        node_feat = torch.cat([inputs, node_attr_tensor], dim=1)

    edge_attr_tensor = torch.from_numpy(edge_attr_array).to(torch.float32)


    device = torch.device('cpu')
    if use_cuda:
        node_feat = node_feat.cuda()
        edge_attr_tensor = edge_attr_tensor.cuda()
        if args.gpuid == 0:
            device = torch.device("cuda:0")
        elif args.gpuid == 1:
            device = torch.device("cuda:1")
        elif args.gpuid == 2:
            device = torch.device("cuda:2")
        elif args.gpuid == 3:
            device = torch.device("cuda:3")
        G = G.to(device)

        train_labeled_nodes_cpu = train_labeled_nodes
        val_labeled_nodes_cpu = val_labeled_nodes
        test_labeled_nodes_cpu = test_labeled_nodes

        train_labeled_nodes = train_labeled_nodes.cuda()
        lat_train_labels = lat_train_labels.cuda()
        lon_train_labels = lon_train_labels.cuda()
        val_labeled_nodes = val_labeled_nodes.cuda()
        lat_val_labels = lat_val_labels.cuda()
        lon_val_labels = lon_val_labels.cuda()
        test_labeled_nodes = test_labeled_nodes.cuda()
        lat_test_labels = lat_test_labels.cuda()
        lon_test_labels = lon_test_labels.cuda()

        all_target_nodeids_tensor = all_target_nodeids_tensor.cuda()
        all_target_domain_labels = all_target_domain_labels.cuda()
    else:
        train_labeled_nodes_cpu = train_labeled_nodes
        val_labeled_nodes_cpu = val_labeled_nodes
        test_labeled_nodes_cpu = test_labeled_nodes

    cur_best_pre_0 = np.inf
    best_epoch = -1
    best_test_dis_list = []
    best_test_nodeid_list = []
    best_test_reachable_rmse = 0
    best_test_unreachable_rmse = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)

    def get_lambda(epoch, max_epochs, max_val):
        p = epoch / max_epochs
        return max_val * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

    for epoch in range(epochs):
        net.train()
        lambda_val = get_lambda(epoch, epochs, args.domain_weight)

        logits, domain_pred = net(G, node_feat, edge_attr_tensor, lambda_val)
        y1_predict = logits[:, 0]
        y2_predict = logits[:, 1]


        out_feat_for_mmd = None
        if use_mmd:
            out_feat_for_mmd = torch.relu(net.lin0(node_feat))
            for i in range(net.num_step_message_passing):
                out_feat_for_mmd = torch.relu(net.conv(G, out_feat_for_mmd, edge_attr_tensor))
                out_feat_for_mmd = net.gnn_dropout(out_feat_for_mmd)

        lat_train_target = lat_train_labels.squeeze(-1)
        lon_train_target = lon_train_labels.squeeze(-1)

        if args.label_smoothing > 0 and net.training:
            noise_lat = torch.randn_like(lat_train_target) * args.label_smoothing
            noise_lon = torch.randn_like(lon_train_target) * args.label_smoothing
            lat_train_target = torch.clamp(lat_train_target + noise_lat, 0, 1)
            lon_train_target = torch.clamp(lon_train_target + noise_lon, 0, 1)

        y1_train_loss = loss_fn(y1_predict[train_labeled_nodes], lat_train_target)
        y2_train_loss = loss_fn(y2_predict[train_labeled_nodes], lon_train_target)
        total_loss = loss1_weight * y1_train_loss + y2_train_loss


        domain_loss = torch.tensor(0.0, device=device)
        if use_dann and domain_pred is not None:
            target_domain_pred = domain_pred[all_target_nodeids_tensor].squeeze(-1)
            domain_loss = domain_loss_fn(target_domain_pred, all_target_domain_labels)

        mmd_loss = torch.tensor(0.0, device=device)
        if use_mmd and out_feat_for_mmd is not None:
            reachable_mask = (all_target_domain_labels > 0.5)
            unreachable_mask = (all_target_domain_labels < 0.5)

            if reachable_mask.sum() > 1 and unreachable_mask.sum() > 1:
                max_samples = 300
                r_indices = all_target_nodeids_tensor[reachable_mask]
                u_indices = all_target_nodeids_tensor[unreachable_mask]
                if len(r_indices) > max_samples:
                    perm = torch.randperm(len(r_indices))[:max_samples]
                    r_indices = r_indices[perm]
                if len(u_indices) > max_samples:
                    perm = torch.randperm(len(u_indices))[:max_samples]
                    u_indices = u_indices[perm]

                r_feat = out_feat_for_mmd[r_indices]
                u_feat = out_feat_for_mmd[u_indices]
                
                feat_std = max(r_feat.std(), u_feat.std())
                sigma = max(feat_std, 0.1)
                
                mmd_loss = compute_mmd(r_feat, u_feat, kernel='rbf', sigma=sigma)
                
                if epoch % 10 == 0:
                    print(f"  [MMD Debug] feat_std={feat_std:.4f}, sigma={sigma:.4f}, mmd_loss={mmd_loss.item():.6f}")

        final_loss = total_loss + args.domain_weight * domain_loss + args.mmd_weight * mmd_loss

        scheduler.step(total_loss)
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            logits_eval, _ = net(G, node_feat, edge_attr_tensor, lambda_val)
            y1_predict_eval = logits_eval[:, 0]
            y2_predict_eval = logits_eval[:, 1]

        temp_y1_predict = y1_predict_eval.detach().cpu().numpy()
        temp_y2_predict = y2_predict_eval.detach().cpu().numpy()
        old_y1_predict = target_scaler1.inverse_transform(temp_y1_predict.reshape(-1, 1))
        old_y2_predict = target_scaler2.inverse_transform(temp_y2_predict.reshape(-1, 1))

        with torch.no_grad():
            if use_cuda:
                net.cpu()

            y1_val_loss = loss_fn(y1_predict_eval[val_labeled_nodes], lat_val_labels.squeeze(-1))
            y2_val_loss = loss_fn(y2_predict_eval[val_labeled_nodes], lon_val_labels.squeeze(-1))
            total_val_loss = loss1_weight * y1_val_loss + y2_val_loss

            sum_dis = 0
            sum_squared_dis = 0
            dis_list = []
            for i, temp in enumerate(old_lat_val_labels):
                temp_lat_label = temp.item()
                temp_lon_label = old_lon_val_labels[i].item()
                temp_lat_pre = old_y1_predict[val_labeled_nodes_cpu][i].item()
                temp_lon_pre = old_y2_predict[val_labeled_nodes_cpu][i].item()
                temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)
                sum_dis += temp_dis
                sum_squared_dis += temp_dis * temp_dis
                dis_list.append(temp_dis)

            error_thisepoch_val = sum_dis / len(old_lat_val_labels)
            mediannum_dis_val = mediannum(dis_list)
            rmse_val = math.sqrt(sum_squared_dis / len(old_lat_val_labels))

            perf_str = ('ValEpoch:%d:avg_error:%.5f:median_error:%.5f:rmse_error:%.5f:'
                        'domain_loss:%.5f:mmd_loss:%.5f\n') % (
                epoch, error_thisepoch_val, mediannum_dis_val, rmse_val,
                domain_loss.item(), mmd_loss.item())
            if print_flag > 0:
                print(perf_str)
            med_err_res_eval.append(mediannum_dis_val)
            ave_err_res_eval.append(error_thisepoch_val)
            rmse_err_res_eval.append(rmse_val)

            if error_thisepoch_val < cur_best_pre_0:
                cur_best_pre_0 = error_thisepoch_val
                best_epoch = epoch
                best_test_dis_list = []
                best_test_nodeid_list = []
                for i, temp in enumerate(old_lat_test_labels):
                    temp_lat_label = temp.item()
                    temp_lon_label = old_lon_test_labels[i].item()
                    temp_lat_pre = old_y1_predict[test_labeled_nodes_cpu][i].item()
                    temp_lon_pre = old_y2_predict[test_labeled_nodes_cpu][i].item()
                    temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)
                    best_test_dis_list.append(temp_dis)
                    best_test_nodeid_list.append(test_nodeid_list[i])
                
                best_test_r_dis = []
                best_test_u_dis = []
                for i, nodeid in enumerate(best_test_nodeid_list):
                    if nodeid_reachable.get(nodeid, 0) == 1:
                        best_test_r_dis.append(best_test_dis_list[i])
                    else:
                        best_test_u_dis.append(best_test_dis_list[i])
                
                best_test_reachable_rmse = math.sqrt(sum([e*e for e in best_test_r_dis]) / len(best_test_r_dis)) if best_test_r_dis else 0
                best_test_unreachable_rmse = math.sqrt(sum([e*e for e in best_test_u_dis]) / len(best_test_u_dis)) if best_test_u_dis else 0

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                error_thisepoch_val, cur_best_pre_0, stopping_step,
                expected_order='dec', flag_step=200)

            sum_dis = 0
            sum_squared_dis = 0
            dis_list = []
            test_r_dis_list, test_u_dis_list = [], []
            for i, temp in enumerate(old_lat_test_labels):
                temp_lat_label = temp.item()
                temp_lon_label = old_lon_test_labels[i].item()
                temp_lat_pre = old_y1_predict[test_labeled_nodes_cpu][i].item()
                temp_lon_pre = old_y2_predict[test_labeled_nodes_cpu][i].item()
                temp_dis = geodistance(temp_lon_label, temp_lat_label, temp_lon_pre, temp_lat_pre)
                sum_dis += temp_dis
                sum_squared_dis += temp_dis * temp_dis
                dis_list.append(temp_dis)
                if test_reachable_list[i] == 1:
                    test_r_dis_list.append(temp_dis)
                else:
                    test_u_dis_list.append(temp_dis)

            error_thisepoch_test = sum_dis / len(old_lat_test_labels)
            mediannum_dis_test = mediannum(dis_list)
            rmse_test = math.sqrt(sum_squared_dis / len(old_lat_test_labels))

            test_r_avg = sum(test_r_dis_list) / len(test_r_dis_list) if test_r_dis_list else 0
            test_u_avg = sum(test_u_dis_list) / len(test_u_dis_list) if test_u_dis_list else 0
            test_r_med = mediannum(test_r_dis_list) if test_r_dis_list else 0
            test_u_med = mediannum(test_u_dis_list) if test_u_dis_list else 0

            test_r_rmse = math.sqrt(sum([e*e for e in test_r_dis_list]) / len(test_r_dis_list)) if test_r_dis_list else 0
            test_u_rmse = math.sqrt(sum([e*e for e in test_u_dis_list]) / len(test_u_dis_list)) if test_u_dis_list else 0

            perf_str = ('TestEpoch:%d:avg_error:%.5f:median_error:%.5f:rmse_error:%.5f:'
                        'reachable_avg:%.5f:reachable_med:%.5f:reachable_rmse:%.5f:'
                        'unreachable_avg:%.5f:unreachable_med:%.5f:unreachable_rmse:%.5f\n') % (
                epoch, error_thisepoch_test, mediannum_dis_test, rmse_test,
                test_r_avg, test_r_med, test_r_rmse,
                test_u_avg, test_u_med, test_u_rmse)
            med_err_res.append(mediannum_dis_test)
            ave_err_res.append(error_thisepoch_test)
            rmse_err_res.append(rmse_test)

            if print_flag > 0:
                print(perf_str)
            if use_cuda:
                net.cuda()

            if should_stop:
                break


    min_err_eval, min_err1_eval, min_err2_eval, ind = get_min_err(
        ave_err_res_eval, med_err_res_eval, rmse_err_res_eval)
    min_err, min_err1, min_err2 = ave_err_res[ind], med_err_res[ind], rmse_err_res[ind]

    print('ave medium err:%.5f' % min_err)
    print('med medium err:%.5f' % min_err1)
    print('rmse err:%.5f' % min_err2)
    print('eval_ave err:%.5f' % min_err_eval)
    print('eval_med err:%.5f' % min_err1_eval)
    print('eval_rmse err:%.5f' % min_err2_eval)

    fw1.write("eval\n")
    for i in ave_err_res_eval:
        fw1.write(str(i) + ' ')
    fw1.write('\n')

    fw1.write("test:\n")
    for i in ave_err_res:
        fw1.write(str(i) + ' ')
    fw1.write('\n')

    fw1.write('eval_ave_err:' + str(min_err_eval) + '\n')
    fw1.write('eval_med_err:' + str(min_err1_eval) + '\n')
    fw1.write('eval_rmse_err:' + str(min_err2_eval) + '\n')
    fw1.write('ave_err:' + str(min_err) + '\n')
    fw1.write('med_err:' + str(min_err1) + '\n')
    fw1.write('rmse_err:' + str(min_err2) + '\n')

    fw1.write('\n========== Best Val Epoch Test Errors ==========\n')
    fw1.write('best_epoch: %d\n' % best_epoch)
    fw1.write('best_val_avg_error: %.5f\n' % cur_best_pre_0)
    fw1.write('test_ip_count: %d\n' % len(best_test_dis_list))
    
    fw1.write('test_errors_list (each IP error in km, format: nodeid:error):\n')
    for i, (nodeid, err) in enumerate(zip(best_test_nodeid_list, best_test_dis_list)):
        fw1.write('%d:%.3f' % (nodeid, err))
        if i < len(best_test_dis_list) - 1:
            fw1.write(' ')
    fw1.write('\n')
    
    fw1.write('test_errors_only (km):\n')
    for i, err in enumerate(best_test_dis_list):
        fw1.write('%.3f' % err)
        if i < len(best_test_dis_list) - 1:
            fw1.write(' ')
    fw1.write('\n')
    
    if best_test_dis_list:
        best_test_avg = sum(best_test_dis_list) / len(best_test_dis_list)
        best_test_med = mediannum(best_test_dis_list)
        best_test_rmse = math.sqrt(sum([e*e for e in best_test_dis_list]) / len(best_test_dis_list))
        fw1.write('best_test_avg_error: %.5f\n' % best_test_avg)
        fw1.write('best_test_med_error: %.5f\n' % best_test_med)
        fw1.write('best_test_rmse_error: %.5f\n' % best_test_rmse)
        
        best_test_r_dis = []
        best_test_u_dis = []
        for i, nodeid in enumerate(best_test_nodeid_list):
            if nodeid_reachable.get(nodeid, 0) == 1:
                best_test_r_dis.append(best_test_dis_list[i])
            else:
                best_test_u_dis.append(best_test_dis_list[i])
        
        if best_test_r_dis:
            fw1.write('best_test_reachable_count: %d\n' % len(best_test_r_dis))
            fw1.write('best_test_reachable_avg: %.5f\n' % (sum(best_test_r_dis) / len(best_test_r_dis)))
            fw1.write('best_test_reachable_med: %.5f\n' % mediannum(best_test_r_dis))
            fw1.write('best_test_reachable_rmse: %.5f\n' % best_test_reachable_rmse)
        if best_test_u_dis:
            fw1.write('best_test_unreachable_count: %d\n' % len(best_test_u_dis))
            fw1.write('best_test_unreachable_avg: %.5f\n' % (sum(best_test_u_dis) / len(best_test_u_dis)))
            fw1.write('best_test_unreachable_med: %.5f\n' % mediannum(best_test_u_dis))
            fw1.write('best_test_unreachable_rmse: %.5f\n' % best_test_unreachable_rmse)
        
        print('\n========== Best Val Epoch Test Errors ==========')
        print('best_epoch: %d' % best_epoch)
        print('best_test_avg_error: %.5f' % best_test_avg)
        print('best_test_med_error: %.5f' % best_test_med)
        print('best_test_rmse_error: %.5f' % best_test_rmse)
        if best_test_r_dis:
            print('best_test_reachable_count: %d' % len(best_test_r_dis))
            print('best_test_reachable_avg: %.5f' % (sum(best_test_r_dis) / len(best_test_r_dis)))
            print('best_test_reachable_med: %.5f' % mediannum(best_test_r_dis))
            print('best_test_reachable_rmse: %.5f' % best_test_reachable_rmse)
        if best_test_u_dis:
            print('best_test_unreachable_count: %d' % len(best_test_u_dis))
            print('best_test_unreachable_avg: %.5f' % (sum(best_test_u_dis) / len(best_test_u_dis)))
            print('best_test_unreachable_med: %.5f' % mediannum(best_test_u_dis))
            print('best_test_unreachable_rmse: %.5f' % best_test_unreachable_rmse)

    fw1.close()
