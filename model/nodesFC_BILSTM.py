import torch
from torch import nn
from model.cells import GRUCell
from torch.nn import Sequential, Linear, Sigmoid


class nodesFC_BILSTM(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device):
        super(nodesFC_BILSTM, self).__init__()
        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size
        self.in_dim = in_dim
        self.hid_dim = 32
        self.num_layers = 2
        self.out_dim = 1
        self.graph_mlp_out = 1
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)
        self.lstm_cell = nn.LSTM(input_size=self.in_dim + self.gnn_out, hidden_size=self.hid_dim, batch_first=True,
                                 num_layers=self.num_layers, bidirectional=True)
        self.gru_cell = GRUCell(self.in_dim + self.graph_mlp_out, self.hid_dim)
        self.graph_mlp = Sequential(Linear(self.city_num * self.in_dim, self.city_num * self.graph_mlp_out),
                                   Sigmoid())

    def forward(self, rain_hist, feature):
        rain_pred = []
        h0 = torch.randn(self.num_layers * 2, self.batch_size, self.hid_dim).to(self.device)
        hn = h0
        c0 = torch.randn(self.num_layers * 2, self.batch_size, self.hid_dim).to(self.device)
        cn = c0
        xn = rain_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len+i]), dim=-1)
            # nodes FC
            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = xn_gnn.view(self.batch_size, -1)
            xn_gnn = self.graph_mlp(xn_gnn)
            xn_gnn = xn_gnn.view(self.batch_size, self.city_num, 1)
            x = torch.cat([xn_gnn, x], dim=-1)
            # nodes FC
            hn = self.gru_cell(x, hn)
            xn, (hn, cn) = self.lstm_cell(x, (hn, cn))
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            rain_pred.append(xn)

        rain_pred = torch.stack(rain_pred, dim=1)
        return rain_pred
