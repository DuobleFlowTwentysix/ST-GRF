import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
import numpy as np
import torch
from collections import OrderedDict
from scipy.spatial import distance
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from geopy.distance import geodesic
from metpy.units import units
import metpy.calc as mpcalc
from bresenham import bresenham

# 读入每个城市的经度纬度
city_fp = os.path.join(proj_dir, 'data/station_id.txt')
# 读入每个城市的海拔高度
altitude_fp = os.path.join(proj_dir, 'data/altitude.npy')


class Graph():
    def __init__(self, use_altitude = True):
        self.dist_thres = 1 # 此处的3是300公里，注意单位
        self.alti_thres = 2500
        self.use_altitude = use_altitude
        
        # 读取每个城市的海拔高度
        self.altitude = self._load_altitude()
        # 读取并记录每个节点
        self.nodes = self._gen_nodes()
        # 将每个节点的海拔高度单独拎出来构成一个node_attr数组
        self.node_attr = self._add_node_attr()
        # 计算节点/城市的数量
        self.node_num = len(self.nodes)
        # edge_index记录了相互影响的城市组
        # edge_attr记录了坐标点之间的方向信息与距离信息
        self.edge_index, self.edge_attr = self._gen_edges()
        # 若要将每个地点的海拔高低信息融入此次计算，则需要运行下面代码
        if self.use_altitude:
            self._update_edges()
        self.edge_num = self.edge_index.shape[1]
        # 将将由边索引和边属性给出的批处理稀疏邻接矩阵转换为单个密集批处理邻接矩阵，矩阵由0和1构成，转换后的矩阵尺寸为[184,184]
        self.adj = to_dense_adj(torch.LongTensor(self.edge_index))[0]
        

    def _load_altitude(self):
        assert os.path.isfile(altitude_fp)
        altitude = np.load(altitude_fp)
        return altitude

    def _lonlat2xy(self, lon, lat, is_aliti):
        if is_aliti:
            lon_l = 100.0
            lon_r = 128.0
            lat_u = 48.0
            lat_d = 16.0
            res = 0.05
        else:
            lon_l = 103.0
            lon_r = 122.0
            lat_u = 42.0
            lat_d = 28.0
            res = 0.125
        x = np.int64(np.round((lon - lon_l - res / 2) / res))
        y = np.int64(np.round((lat_u + res / 2 - lat) / res))
        return x, y

    def _gen_nodes(self):
        # 创建一个有序字典
        nodes = OrderedDict()
        # 将节点装入有序字典中
        with open(city_fp, 'r') as f:
            for line in f:
                idx, city, lon, lat = line.rstrip('\n').split('\t')
                idx = int(idx)
                lon, lat = float(lon), float(lat)
                x, y = self._lonlat2xy(lon, lat, True)
                altitude = self.altitude[y, x]
                # idx为字典中的键，字典{'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}为该字典中的值
                nodes.update({idx: {'city': city, 'altitude': altitude, 'lon': lon, 'lat': lat}})
        return nodes

    def _add_node_attr(self):
        node_attr = []
        altitude_arr = []
        for i in self.nodes:
            altitude = self.nodes[i]['altitude']
            altitude_arr.append(altitude)
        altitude_arr = np.stack(altitude_arr)
        node_attr = np.stack([altitude_arr], axis=-1)
        return node_attr

    def traverse_graph(self):
        lons = []
        lats = []
        citys = []
        idx = []
        for i in self.nodes:
            idx.append(i)
            city = self.nodes[i]['city']
            lon, lat = self.nodes[i]['lon'], self.nodes[i]['lat']
            lons.append(lon)
            lats.append(lat)
            citys.append(city)
        return idx, citys, lons, lats

    def gen_lines(self):

        lines = []
        for i in range(self.edge_index.shape[1]):
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            lines.append(([src_lon, dest_lon], [src_lat, dest_lat]))

        return lines

    def _gen_edges(self):
        coords = []
        lonlat = {}
        for i in self.nodes:
            coords.append([self.nodes[i]['lon'], self.nodes[i]['lat']])
        # 求coords中的坐标点之间的距离，距离的类型为'eudlidean'
        dist = distance.cdist(coords, coords, 'euclidean')
        # 创建数组adj来记录坐标点之间pm2.5互相传播的情况
        adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        # 距离小于阈值dist_thres时，表明两个坐标点之间会相互影响，记为1
        adj[dist <= self.dist_thres] = 1
        # 断言，判断adj的形状与dist的形状是否相同
        assert adj.shape == dist.shape
        # 若两个数据点之间会出现pm2.5的传播，则该矩阵记录两个数据点之间的距离；若两个数据点之间不存在pm2.5的传播，则两个数据点之间为0
        dist = dist * adj
        # 将密集矩阵转化为稀疏矩阵
        edge_index, dist = dense_to_sparse(torch.tensor(dist))
        # 从张量转为矩阵形式
        edge_index, dist = edge_index.numpy(), dist.numpy()
        
        direc_arr = []
        dist_kilometer = []
        # edge_index记录了相互影响的城市组，edge_index的记录形式为：
        # [[  0   0   0 ... 183 183 183]
        # [  1   2   3 ... 179 181 182]]
        # 共有3938个相互影响的城市组
        for i in range(edge_index.shape[1]):
            # src是pm2.5传播的源头城市，dest是pm2.5传播的目标城市
            # edge_index第一行是影响起源的城市，edge_index第二行是受影响的城市
            src, dest = edge_index[0, i], edge_index[1, i]
            # 读取起源城市的经度
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            # 读取目标城市的纬度
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_location = (src_lat, src_lon)
            dest_location = (dest_lat, dest_lon)
            # 计算点之间的测地线距离
            dist_km = geodesic(src_location, dest_location).kilometers
            v, u = src_lat - dest_lat, src_lon - dest_lon
            # MetPt是一种专门用于地理气象计算的库，units.meter表示的是单位“米”，units.second表示的是单位“秒”。
            u = u * units.meter / units.second
            v = v * units.meter / units.second
            # wind_direciton()为计算风的方向专用的函数，此处借助该函数来计算两个城市之间连线与正北方向的角度
            # 计算出的结果是以 [0, 360] 度为间隔的风向，其中 360 为北
            direc = mpcalc.wind_direction(u, v)._magnitude
            direc = round((direc / 360) * 16)
            direc_arr.append(direc)
            # dist_kilometer记录城市组之间的距离
            dist_kilometer.append(dist_km)
        # 创建数组direc_arr用来存放坐标点之间的方向信息，形状为(3938,1)，edge_index中也恰好记录了3938组相互影响的城市组
        direc_arr = np.stack(direc_arr)
        # 创建数组dist_arr用来存放坐标点之间的距离信息，形状为(3938,1)，dege_index中也恰好记录了3938组相互影响的城市组
        dist_arr = np.stack(dist_kilometer)
        # 将坐标点之间的方向信息与距离信息合并成一个列表,形状为(3938,2)
        attr = np.stack([dist_arr, direc_arr], axis=-1)

        return edge_index, attr

    def _update_edges(self):
        edge_index = []
        edge_attr = []
        for i in range(self.edge_index.shape[1]):
            
            src, dest = self.edge_index[0, i], self.edge_index[1, i]
            # 读取起源城市的经度与维度
            src_lat, src_lon = self.nodes[src]['lat'], self.nodes[src]['lon']
            # 读取目标城市的经度与维度
            dest_lat, dest_lon = self.nodes[dest]['lat'], self.nodes[dest]['lon']
            src_x, src_y = self._lonlat2xy(src_lon, src_lat, True)
            dest_x, dest_y = self._lonlat2xy(dest_lon, dest_lat, True)
            points = np.asarray(list(bresenham(src_y, src_x, dest_y, dest_x))).transpose((1,0))
            altitude_points = self.altitude[points[0], points[1]]
            altitude_src = self.altitude[src_y, src_x]
            altitude_dest = self.altitude[dest_y, dest_x]
            if np.sum(altitude_points - altitude_src > self.alti_thres) < 3 and \
               np.sum(altitude_points - altitude_dest > self.alti_thres) < 3:
                edge_index.append(self.edge_index[:,i])
                edge_attr.append(self.edge_attr[i])

        self.edge_index = np.stack(edge_index, axis=1)
        self.edge_attr = np.stack(edge_attr, axis=0)


if __name__ == '__main__':
    graph = Graph()