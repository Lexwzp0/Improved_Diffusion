import os
from scipy import io
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Dataset
import numpy
from scipy import io
from sklearn.model_selection import train_test_split
# from torch_geometric.utils import add_self_loops
import torch.nn.functional as F


class DataToGraph(Dataset):
    def __init__(self, raw_data_path, dataset_name, transform=None, pre_transform=None, pre_filter=None):
        self.raw_data_path = raw_data_path
        self.dataset_name = dataset_name
        super(DataToGraph, self).__init__(raw_data_path, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['NPS_64sensors_13type.mat', 'NPS.mat', 'TFF.mat']

    @property
    def processed_file_names(self):
        return ['process_NPS_64sensors_13type.pt', 'process_NPS.pt', 'process_TFF.pt']

    def process(self):
        # 将数据处理为图列表和标签列表
        data = io.loadmat(os.path.join(self.raw_data_path, self.dataset_name))
        if self.dataset_name == 'NPS_64sensors_13type.mat' or self.dataset_name == 'TFF.mat':
            x_set = numpy.concatenate((data['x_train'], data['x_test']), axis=2)  # (D x V x N)
            labels = numpy.concatenate((data['y_train'], data['y_test']), axis=0)
        elif self.dataset_name == 'NPS.mat':
            x_set = data['data_x']  # (21918 x 50 x 122)
            x_set = x_set.swapaxes(0, 2)  # (122 x 50 x 21918)
            x_set = x_set.swapaxes(0, 1)  # (50 x 122 x 21918)
            labels = data['data_y'].reshape(-1, 1)
        else:
            raise Exception('没有该数据集！！！')

        self.num_tasks = 1
        self.eval_metric = 'ROC-AUC'
        self.task_type = 'classification'
        self.num_classes1 = int(labels.max()) - int(labels.min()) + 1
        self.binary = True
        self.num_nodes = x_set.shape[1]

        labels = np.squeeze(np.array(labels))
        if self.dataset_name == 'NPS_64sensors_13type.mat' or self.dataset_name == 'NPS_64sensors_13type.mat':
            labels = labels - 1  # TODO 阴差阳错写错了也没影响，实际就只有NPS_64sensors_13type的label是从1开始！！！
        labels = torch.tensor(labels, dtype=torch.long)
        # labels = F.one_hot(labels, num_classes=self.num_classes)  # (N x num_classes)
        # TODO 图是全连接，如果边不动态学习没有任何意义
        init_adj = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.float) - torch.eye(self.num_nodes,
                                                                                               dtype=torch.float)
        edge_index, _ = dense_to_sparse(init_adj)        # edge_index = init_adj.to_sparse_coo().indices()  # (2 x E)

        e_feat = init_adj.to_sparse_coo().values().unsqueeze(-1)  # (E x 1)
        e_feat = e_feat.float()  # TODO 这里要是float的
        # edge_index, _ = dense_to_sparse(init_adj)
        # print('edge_index:', edge_index)  # (2 x E) = (2 x V*V-V)
        graph_list = []

        for k in range(labels.size(0)):
            x = x_set[:, :, k]
            x = torch.tensor(x, dtype=torch.float)
            x = torch.transpose(x, 1, 0)
            # print('x.shape:', x.shape)  # (V x d)
            # print('labels[k]:', labels[k])
            g = Data(x=x, edge_index=edge_index, edge_attr=e_feat, y=labels[k].unsqueeze(0))
            graph_list.append(g)
        self.graphs = graph_list
        self.labels = labels
        # return self.graphs

    def get_idx_split(self, seed=2, train_ratio=0.8, val_ratio=0.1):  # TODO 自定义数据集划分, train/test/val = 80%/10%/10%
        index = list(range(len(self.labels)))
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio
        # train,val,test划分
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, self.labels[index], stratify=self.labels[index],
                                                                train_size=self.train_ratio, random_state=seed, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=self.test_ratio / (
                                                                        self.val_ratio + self.test_ratio),
                                                                random_state=seed, shuffle=True)
        # TODO (num_train,)
        return {'train': torch.tensor(idx_train, dtype=torch.long), 'valid': torch.tensor(idx_valid, dtype=torch.long),
                'test': torch.tensor(idx_test, dtype=torch.long)}

    def len(self):
        """数据集中图的数量"""
        return len(self.graphs)

    def get(self, idx):
        """ 通过idx获取对应的图和标签"""
        return self.graphs[idx]
