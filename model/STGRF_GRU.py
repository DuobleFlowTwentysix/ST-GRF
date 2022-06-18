import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid
import numpy as np
from torch_scatter import scatter_add# scatter_sub  # no scatter sub in lastest PyG
from torch.nn import functional as F
from torch.nn import Parameter


class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        # edge_index中记录了相互影响的城市组，每当有两个城市相互影响，就会在知识图谱中出现一条边
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        # 边的属性，edge_attr记录了相互影响的城市组之前的方向信息与距离信息
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        # 正则化处理后的边属性
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(dim=0)
        # 随机赋予权重参数
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        # 风速的平均值与方差
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        e_h = 32
        e_out = 30
        n_out = out_dim
        # Sequential()可以看作一个容器，用来包装各层的模型
        # 模型输入的维度是in_dim*2+2+1呢
        self.edge_mlp = Sequential(Linear(in_dim * 2 + 2 + 1, e_h),
                                   Sigmoid(),
                                   Linear(e_h, e_out),
                                   Sigmoid(),
                                   )
        self.node_mlp = Sequential(Linear(e_out, n_out),
                                   Sigmoid(),
                                   )

    def forward(self, x):
        # edge_index中记录了相互污染的城市组
        self.edge_index = self.edge_index.to(self.device)
        # edge_attr中记录了相互污染的城市组之间的距离和直线距离的角度
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)
        # 输入数据x的维度为[16, 184, 13]，是由特征因子和历史pm2.5浓度数据拼接而成的数据
        edge_src, edge_target = self.edge_index
        # 读出污染起源城市的特征因子值与pm2.5浓度值
        node_src = x[:, edge_src]
        # 读出被污染城市的特征因子值与pm2.5浓度值
        node_target = x[:, edge_target]
        # 根据输入的特征因子数据计算起源城市的风速与风的方向
        src_wind = node_src[:,:,-2:] * self.wind_std[None,None,:] + self.wind_mean[None,None,:]
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:,:,1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:,:,0]
        city_direc = self.edge_attr_[:,:,1]
        # city_direc是两个城市连线的方向与正北方向的夹角
        # src_wind_direc是风速方向与正北方向的夹角
        theta = torch.abs(city_direc - src_wind_direc)
        # 计算论文中的平流系数S，用来表示风源节点对汇聚节点的影响程度
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta * 360 / 16) / city_dist)
        edge_weight = edge_weight.to(self.device)
        # 进行正则化之后的边的属性，其中包括正则化之后的城市间距离和城市连线的角度
        edge_attr_norm = self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        # out的维度为13+13+2+1
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:,:,None]], dim=-1)

        out = self.edge_mlp(out)
        # 对输出out根据edge_target和edge_src进行归约操作
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(out.neg(), edge_src, dim=1, dim_size=x.size(1))  # For higher version of PyG.

        out = out_add + out_sub
        # 返回一个用来表示节点的mlp网络
        out = self.node_mlp(out)

        return out

# in_dim=9;city_num=491;batch_size=16
class STGRF_GRU(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device, edge_index, edge_attr, wind_mean, wind_std):
        super(STGRF_GRU, self).__init__()

        self.device = device
        self.hist_len = hist_len#1
        self.pred_len = pred_len#24
        self.city_num = city_num
        self.batch_size = batch_size#32

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 9
        #self.gnn_out = 13

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std)
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, rain_hist, feature):
        rain_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = rain_hist[:, -1]
        for i in range(self.pred_len):
            # 根据时序的进行，特征的输入会随着时间更新
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            # 将数据输入GNN模块
            xn_gnn = self.graph_gnn(xn_gnn)
            # GNN模块的输出与原来的输入数据进行拼接作为gru的输入
            x = torch.cat([xn_gnn, x], dim=-1)
            # 根据gru网络对参数hn进行更新
            hn = self.gru_cell(x, hn)
            # 返回一个有相同数据但不同大小的Tensor。
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            # 最后一层为全连接层，调整维度后输出结果
            xn = self.fc_out(xn)
            rain_pred.append(xn)

        rain_pred = torch.stack(rain_pred, dim=1)

        return rain_pred
