import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)
from util import config, file_dir

from datetime import datetime
import numpy as np
import arrow
import metpy.calc as mpcalc
from metpy.units import units
from torch.utils import data


class HazeData(data.Dataset):

    def __init__(self, graph,
                       hist_len=1,
                       pred_len=1,
                       dataset_num=1,
                       flag='Train',
                       ):

        if flag == 'Train':
            start_time_str = 'train_start'
            end_time_str = 'train_end'
        elif flag == 'Val':
            start_time_str = 'val_start'
            end_time_str = 'val_end'
        elif flag == 'Test':
            start_time_str = 'test_start'
            end_time_str = 'test_end'
        else:
            raise Exception('Wrong Flag!')
        # 用于训练模型的数据的开始时间与结束时间，论文中给出了三种分割数据集的方法
        self.start_time = self._get_time(config['dataset'][dataset_num][start_time_str])
        self.end_time = self._get_time(config['dataset'][dataset_num][end_time_str])
        # data_start是数据集整体的开始时间，即2015, 1, 1
        # data_end是数据集的整体的结束时间，即2018, 12, 31
        self.data_start = self._get_time(config['dataset']['data_start'])
        self.data_end = self._get_time(config['dataset']['data_end'])

        self.rainfall_fp = file_dir['rainfall_fp']

        self.graph = graph
        # 读取knowair数据集,读出标签和特征因子等数据
        self._load_npy()
        # 返回从2015年1月1日到2018年12月31日的时间戳，并存放在time_arr中
        self._gen_time_arr()
        # 对记录时间戳的数组time_arr和记录时间的数组time_arrow进行修正
        self._process_time()
        # 挑选8个特征因子,在这8个特征因子的基础上新增了4个特征因子——当前的小时数、当前星期数、风速与风的方向
        self._process_feature()
        self.feature = np.float32(self.feature)
        self.rain = np.float32(self.rain)
        # 求特征值与pm2.5在时间上的平均值与方差
        self._calc_mean_std()
        # seq_len=25
        seq_len = hist_len + pred_len
        self._add_time_dim(seq_len)
        self._norm()

    def _norm(self):
        self.feature = (self.feature - self.feature_mean) / self.feature_std
        self.rain = (self.rain - self.rain_mean) / self.rain_std
    # 历史数据时序长度hist_len是1，预测数据时序长度pred_len是24，故一次预测的总时序长度为25
    # 该函数根据时序长度对数据进行长度为25封装，[0:25]、[1:26]、[2:27]....
    def _add_time_dim(self, seq_len):

        def _add_t(arr, seq_len):
            t_len = arr.shape[0]
            assert t_len > seq_len
            arr_ts = []
            seq_len = int(seq_len)
            t_len = int(t_len)
            for i in range(seq_len, t_len):
                arr_t = arr[i-seq_len:i]
                arr_ts.append(arr_t)
            arr_ts = np.stack(arr_ts, axis=0)
            return arr_ts

        self.rain = _add_t(self.rain, seq_len)
        self.feature = _add_t(self.feature, seq_len)
        self.time_arr = _add_t(self.time_arr, seq_len)

    def _calc_mean_std(self):
        # 对每个特征因子求平均值和方差
        self.feature_mean = self.feature.mean(axis=(0,1))
        self.feature_std = self.feature.std(axis=(0,1))
        # 特征值的最后两项是风力相关的特征值的平均值与方差
        self.wind_mean = self.feature_mean[-2:]
        self.wind_std = self.feature_std[-2:]
        # 对pm2.5浓度求平均值与方差
        self.rain_mean = self.rain.mean()
        self.rain_std = self.rain.std()

    def _process_feature(self):
        # 从17个特征因子中挑选8个特征因子用作训练，特征因子的名称详细见config.yaml
        #metero_var = config['data']['metero_var']
        #metero_use = config['experiments']['metero_use']
        #metero_idx = [metero_var.index(var) for var in metero_use]
        # 读取背挑选的8个特征因子的数据
        #self.feature = self.feature[:,:,metero_idx]
        # 风的东西方向数据
        #u = self.feature[:, :, -2] * units.meter / units.second
        # 风的南北方向数据
        #v = self.feature[:, :, -1] * units.meter / units.second
        # 求出风速
        #speed = 3.6 * mpcalc.wind_speed(u, v)._magnitude
        # 求成风的方向
        #direc = mpcalc.wind_direction(u, v)._magnitude
        speed = self.feature[:, :, -2] * 3.6
        direc = self.feature[:, :, -1]
        h_arr = []
        w_arr = []
        for i in self.time_arrow:
            #h_arr.append(i.hour)
            # 返回今天是周几
            w_arr.append(i.month)
        #h_arr = np.stack(h_arr, axis=-1)
        w_arr = np.stack(w_arr, axis=-1)
        # node_num中存放了城市的数量，np.repeat是复制粘贴的作用，axis=1表示以增加列的形式进行复制粘贴
        #h_arr = np.repeat(h_arr[:, None], self.graph.node_num, axis=1)
        w_arr = np.repeat(w_arr[:, None], self.graph.node_num, axis=1)
        self.feature = np.concatenate([self.feature[:, :, :-2], w_arr[:, :, None], speed[:, :, None], direc[:, :, None]], axis=-1)


    def _process_time(self):
        start_idx = self._get_idx(self.start_time)
        end_idx = self._get_idx(self.end_time)
        # 创建二维数组存放pm2.5浓度，二维数组的行数等于时间戳的数量
        self.rain = self.rain[start_idx: end_idx+1, :]
        # 创建二维数组存放特征，二维数组的行数等于时间戳的数量
        self.feature = self.feature[start_idx: end_idx+1, :]
        self.time_arr = self.time_arr[start_idx: end_idx+1]
        self.time_arrow = self.time_arrow[start_idx: end_idx + 1]
    # 返回从2015年1月1日到2018年12月31日的每3个小时一次的时间戳
    def _gen_time_arr(self):
        self.time_arrow = []
        self.time_arr = []
        for time_arrow in arrow.Arrow.interval('hour', self.data_start, self.data_end.shift(hours=+24), 24):
            self.time_arrow.append(time_arrow[0])
            # 获取当前的时间戳
            self.time_arr.append(time_arrow[0].timestamp)
        self.time_arr = np.stack(self.time_arr, axis=-1)

    def _load_npy(self):
        # knowair是一个三维数组，维度为(11688, 184, 18)，三个维度分别表示时间、城市、特征因子
        # 数据集是以3小时为间隔进行测试的
        self.rainfall = np.load(self.rainfall_fp)
        # 第三个维度有18个变量，此处取前面17个变量作为预测的特征因子
        self.feature = self.rainfall[:,:,:-1]
        # 第三个维度的最后一个变量则是标签
        self.rain = self.rainfall[:,:,-1:]
    # 根据先后顺序给时间戳一个序号
    def _get_idx(self, t):
        t0 = self.data_start
        return int((t.timestamp - t0.timestamp) / (60 * 60 * 24))

    def _get_time(self, time_yaml):
        arrow_time = arrow.get(datetime(*time_yaml[0]), time_yaml[1])
        return arrow_time

    def __len__(self):
        return len(self.rain)

    def __getitem__(self, index):
        return self.rain[index], self.feature[index], self.time_arr[index]

if __name__ == '__main__':
    from graph import Graph
    graph = Graph()
    train_data = HazeData(graph, flag='Train')
    val_data = HazeData(graph, flag='Val')
    test_data = HazeData(graph, flag='Test')

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
