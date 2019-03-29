# -*- coding: utf-8 -*-
from ..core import *
from ..autograd import *
from ..functions import listconcat

class RNNCell(Function):
    @staticmethod
    def forward(x, h, weight_x, weight_h, bias_x, bias_h):
        '''
        Shape:
            - x: [N, input_size]
            - h: [N, hidden_size]
            - weight_x: [input_size, hidden_size]
            - weight_h: [hidden_size, hidden_size]
            - bias_x: [hidden_size]
            - bias_h: [hidden_size]
            - Output: [N, hidden_size]
        '''
        if bias_x is None or bias_h is None:
            tmp = np.add(np.dot(h.data, weight_h.data), np.dot(x.data, weight_x.data))
            result = Tensor(np.tanh(tmp))
            result.set_creator(RNNCell.prepare(result.shape, x, h, weight_x, weight_h, bias=False, tmp=result.data))
        else:
            tmp = np.add(np.add(np.dot(h.data, weight_h.data), bias_x.data), np.add(np.dot(x.data, weight_x.data), bias_h.data))
            result = Tensor(np.tanh(tmp))
            result.set_creator(RNNCell.prepare(result.shape, x, h, weight_x, weight_h, bias_x, bias_h, bias=True, tmp=result.data))
        return result

    def calc_grad(self, dh_next):
        dt = np.multiply(dh_next, np.subtract(1, np.square(self.kwargs['tmp'])))
        dw_x = np.dot(self.var[0].data.T, dt)
        dw_h = np.dot(self.var[1].data.T, dt)
        dx = np.dot(dt, self.var[2].data.T)
        dh = np.dot(dt, self.var[3].data.T)

        if not self.kwargs['bias']:
            return dx, dh, dw_x, dw_h
        else:
            db_x = RNNCell.handle_broadcast(dt, self.var[4])
            db_h = RNNCell.handle_broadcast(dt, self.var[5])
            return dx, dh, dw_x, dw_h, db_x, db_h

rnncell = RNNCell(None)

class RNN(Function):
    @staticmethod
    def forward(x, h, weight_x, weight_h, bias_x, bias_h, num_layers):
        '''
        Shape:
            - x: [seq_len, N, input_size]
            - h: [num_layers, N, hidden_size]
            - Output: [seq_len, N, hidden_size]
            - Hidden: [num_layers, N, hidden_size]
        '''
        seq_len = x.shape[0]
        h_out = []
        tmp = [x]
        for l in range(num_layers):
            hx = h[l]
            tmp.append([])
            for i in range(seq_len):
                if bias_x is None or bias_h is None:
                    hx = rnncell(tmp[l][i], hx, weight_x[l], weight_h[l], None, None)
                    tmp[l+1].append(hx)
                else:
                    hx = rnncell(tmp[l][i], hx, weight_x[l], weight_h[l], bias_x[l], bias_h[l])
                    tmp[l+1].append(hx)
            h_out.append(hx)
        result_x = listconcat(tmp[-1])
        result_h = listconcat(h_out)
        return result_x, result_h

rnn = RNN(None)
