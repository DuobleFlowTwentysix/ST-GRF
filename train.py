import os
import sys
proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
from util import config, file_dir
from graph import Graph
# 详细见dataset.py
from dataset import HazeData

from model.MLP import MLP
from model.LSTM import LSTM
from model.GRU import GRU
from model.GC_LSTM import GC_LSTM
from model.nodesFC_GRU import nodesFC_GRU
from model.STGRF_GRU import STGRF_GRU
from model.STGRF_GRU_nosub import STGRF_GRU_nosub
from model.STGRF_LSTM import STGRF_LSTM
from model.STGRF_BILSTM import STGRF_BILSTM
from model.STGRF_BILSTM_nosub import STGRF_BILSTM_nosub
from model.STGRF_BIGRU import STGRF_BIGRU
import matplotlib.pyplot as plt
import arrow
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
import glob
import shutil
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', dest='model', default='MLP', help='choose model')
parser.add_argument('--pred_len', dest='pred_len', default=1, help='decide pred_len')
parser.add_argument('--dataset_num', dest='dataset_num', default=2, help='decide dataset_num')
args = parser.parse_args()

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# 读取训练数据
graph = Graph()
# city的数量等于图谱中节点的数量
city_num = graph.node_num
# 参数设置详见util.py中的config函数，参数记录在config.yaml中
batch_size = config['train']['batch_size']#32
epochs = config['train']['epochs']#50
hist_len = config['train']['hist_len']#1
#pred_len = config['train']['pred_len']#1
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
# dataset_num = config['experiments']['dataset_num']
# = config['experiments']['model']
exp_model = args.model
pred_len = int(args.pred_len)
dataset_num = int(args.dataset_num)
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
# 求均方误差的内嵌公式
criterion = nn.MSELoss()
# HazeData是雾霾天数据集
train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
print('trainready')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
print('valready')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
print('testready')
in_dim = train_data.feature.shape[-1] + train_data.rain.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
rain_mean, rain_std = test_data.rain_mean, test_data.rain_std


def get_metric(predict_epoch, label_epoch):
    # pm2.5浓度高于75时表明发生雾霾，发出雾霾预警
    heavy_threshold = 25
    moderate_threshold = 10
    light_threshold = 0.1
    predict_light = ((predict_epoch >= light_threshold) & (predict_epoch < moderate_threshold))
    predict_moderate = ((predict_epoch >= moderate_threshold) & (predict_epoch < heavy_threshold))
    predict_heavy = predict_epoch > heavy_threshold
    label_light = ((label_epoch >= light_threshold) & (label_epoch < moderate_threshold))
    label_moderate = ((label_epoch >= moderate_threshold) & (label_epoch < heavy_threshold))
    label_heavy = label_epoch > heavy_threshold

    hit_light = np.sum(np.logical_and(predict_light, label_light))
    miss_light = np.sum(np.logical_and(label_light, predict_moderate+predict_heavy))
    falsealarm_light = np.sum(np.logical_and(predict_light, label_moderate+label_heavy))
    csi_light = hit_light / (hit_light + falsealarm_light + miss_light)
    pod_light = hit_light / (hit_light + miss_light)
    far_light = falsealarm_light / (hit_light + falsealarm_light)

    hit_moderate = np.sum(np.logical_and(predict_moderate, label_moderate))
    miss_moderate = np.sum(np.logical_and(label_moderate, predict_light+predict_heavy))
    falsealarm_moderate = np.sum(np.logical_and(predict_moderate, label_light+label_heavy))
    csi_moderate = hit_moderate / (hit_moderate + falsealarm_moderate + miss_moderate)
    pod_moderate = hit_moderate / (hit_moderate + miss_moderate)
    far_moderate = falsealarm_moderate / (hit_moderate + falsealarm_moderate)

    hit_heavy = np.sum(np.logical_and(predict_heavy, label_heavy))
    miss_heavy = np.sum(np.logical_and(label_heavy, predict_light+predict_moderate))
    falsealarm_heavy = np.sum(np.logical_and(predict_heavy, label_moderate+label_light))
    csi_heavy = hit_heavy / (hit_heavy + falsealarm_heavy + miss_heavy)
    pod_heavy = hit_heavy / (hit_heavy + miss_heavy)
    far_heavy = falsealarm_heavy / (hit_heavy + falsealarm_heavy)

    '''
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    '''
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi_light, pod_light, far_light, csi_moderate, pod_moderate, far_moderate, csi_heavy, pod_heavy, far_heavy

def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'early_stop: %s\n' % early_stop + \
                'lr: %s\n' % lr + \
                '========================================\n'
    return exp_info


def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    elif exp_model == 'LSTM':
        return LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GRU':
        return GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'nodesFC_GRU':
        return nodesFC_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device)
    elif exp_model == 'GC_LSTM':
        return GC_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index)
    elif exp_model == 'STGRF_GRU':
        return STGRF_GRU(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'STGRF_GRU_nosub':
        return STGRF_GRU_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'STGRF_LSTM':
        return STGRF_LSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'STGRF_BILSTM':
        return STGRF_BILSTM(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr,
                           wind_mean, wind_std)
    elif exp_model == 'STGRF_BIGRU':
        return STGRF_BIGRU(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)
    elif exp_model == 'STGRF_BILSTM_nosub':
        return STGRF_BILSTM_nosub(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr, wind_mean, wind_std)

    else:
        raise Exception('Wrong model name!')


def train(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader)):
        rain, feature, time_arr = data
        # pm2.5的尺寸为[16, 25, 184, 1]
        # feature的尺寸为[16, 25, 184, 12]
        # time_arr的尺寸为[16, 25]
        rain = rain.to(device)
        feature = feature.to(device)
        rain_label = rain[:, hist_len:]
        rain_hist = rain[:, :hist_len]
        rain_pred = model(rain_hist, feature)
        # 根据训练结果求均方误差
        loss = criterion(rain_pred, rain_label)
        # 反向传播
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= batch_idx + 1
    return train_loss


def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in tqdm(enumerate(val_loader)):
        rain, feature, time_arr = data
        rain = rain.to(device)
        feature = feature.to(device)
        rain_label = rain[:, hist_len:]
        rain_hist = rain[:, :hist_len]
        rain_pred = model(rain_hist, feature)
        loss = criterion(rain_pred, rain_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        rain, feature, time_arr = data
        rain = rain.to(device)
        feature = feature.to(device)
        rain_label = rain[:, hist_len:]
        # pm25_hist为历史的pm2.5浓度数值，根据历史数据训练模型进而预测未来数据
        rain_hist = rain[:, :hist_len]
        rain_pred = model(rain_hist, feature)
        # 计算mse误差，由此可知，pm25_pred是模型预测的pm2.5值，pm25_label是数据集中pm2.5浓度的标签
        loss = criterion(rain_pred, rain_label)
        test_loss += loss.item()
        # pm25_pred_val是存放pm2.5浓度预测结果的数组
        rain_pred_val = np.concatenate([rain_hist.cpu().detach().numpy(), rain_pred.cpu().detach().numpy()], axis=1) * rain_std + rain_mean
        # pm25_label_val是存放pm2.5浓度预测结果的数组
        rain_label_val = rain.cpu().detach().numpy() * rain_std + rain_mean
        predict_list.append(rain_pred_val)
        label_list.append(rain_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1
    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    # 获取训练相关超参数
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')
    # rmse和mae同为计算预测误差的公式，csi=准测预测数/(准测预测数+未准确预测数+误预测的数量),
    # far=误预测的数量/(准确预测数+误预测的数量)，pod=准确预测数/(准确预测数+未准确预测数)
    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_light_list, pod_light_list, far_light_list, csi_moderate_list, pod_moderate_list, far_moderate_list, csi_heavy_list, pod_heavy_list, far_heavy_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    # exp_repeat为实验重复的次数
    '''
    训练实验注意：
    1、每一次实验，都会训练一个新的模型
    2、每一次实验都会进行50个epoch
    3、在当前实验次数中，若遇到loss比往常的epoch更低的epoch，将会记录当前epoch和loss、rmse、mae、csi、pod和far，并用一个结果表格记录模型效果
    4、实验次数有很多次，但结果表格只有一个，最后对整个结果表格求平均值和方差来表示模型的整体效果。
    '''
    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)
        # 加载数据集
        # torch.utils.data.DataLoader自动拆解了HazeData这个类，使其能输出pm2.5、feature和time_arr
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
        # 加载模型
        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        for epoch in range(epochs):
            print('\nTrain epoch %s:' % (epoch))

            train_loss = train(train_loader, model, optimizer)
            val_loss = val(val_loader, model)

            print('train_loss: %.4f' % train_loss)
            print('val_loss: %.4f' % val_loss)
            # early stop机制
            if epoch - best_epoch > early_stop:
                break
            # 遇到更好的损失值，进行保存
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_epoch = epoch
                print('Minimum val loss!!!')
                # 保存模型
                torch.save(model.state_dict(), model_fp)
                print('Save model: %s' % model_fp)

                test_loss, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
                train_loss_, val_loss_ = train_loss, val_loss
                rmse, mae, csi_light, pod_light, far_light, csi_moderate, pod_moderate, far_moderate, csi_heavy, pod_heavy, far_heavy = get_metric(predict_epoch, label_epoch)
                print('Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f' % (train_loss_, val_loss_, test_loss,))
                if save_npy:
                    np.save(os.path.join(exp_model_dir, 'predict.npy'), predict_epoch)
                    np.save(os.path.join(exp_model_dir, 'label.npy'), label_epoch)
                    np.save(os.path.join(exp_model_dir, 'time.npy'), time_epoch)

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss)
        rmse_list.append(rmse)
        mae_list.append(mae)
        csi_light_list.append(csi_light)
        pod_light_list.append(pod_light)
        far_light_list.append(far_light)
        csi_moderate_list.append(csi_moderate)
        pod_moderate_list.append(pod_moderate)
        far_moderate_list.append(far_moderate)
        csi_heavy_list.append(csi_heavy)
        pod_heavy_list.append(pod_heavy)
        far_heavy_list.append(far_heavy)
        print('\nNo.%2d experiment results:' % exp_idx)
        print(
            'Train loss: %0.4f, Val loss: %0.4f, Test loss: %0.4f' % (
            train_loss_, val_loss_, test_loss))

    exp_metric_str = '---------------------------------------\n' + \
                     'train_loss  | mean: %0.4f std: %0.4f\n' % (get_mean_std(train_loss_list)) + \
                     'val_loss    | mean: %0.4f std: %0.4f\n' % (get_mean_std(val_loss_list)) + \
                     'test_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(test_loss_list)) + \
                     'RMSE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(rmse_list)) + \
                     'MAE         | mean: %0.4f std: %0.4f\n' % (get_mean_std(mae_list)) + \
                     'CSI_light   | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_light_list)) + \
                     'POD_light   | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_light_list)) + \
                     'FAR_light   | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_light_list)) + \
                     'CSI_moderate| mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_moderate_list)) + \
                     'POD_moderate| mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_moderate_list)) + \
                     'FAR_moderate| mean: %0.4f std: %0.4f\n' % (get_mean_std(far_moderate_list)) + \
                     'CSI_heavy   | mean: %0.4f std: %0.4f\n' % (get_mean_std(csi_heavy_list)) + \
                     'POD_heavy   | mean: %0.4f std: %0.4f\n' % (get_mean_std(pod_heavy_list)) + \
                     'FAR_heavy   | mean: %0.4f std: %0.4f\n' % (get_mean_std(far_heavy_list))
    metric_fp = os.path.join(os.path.dirname(exp_model_dir), 'metric.txt')
    with open(metric_fp, 'w') as f:
        f.write(exp_info)
        f.write(str(model))
        f.write(exp_metric_str)

    print('=========================\n')
    print(exp_info)
    print(exp_metric_str)
    print(str(model))
    print(metric_fp)
    print(get_mean_std(rmse_list))


if __name__ == '__main__':
    main()
