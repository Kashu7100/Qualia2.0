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

class LSTMCell(Function):
    def __init__(self, output_shape_h, output_shape_c, *args, **kwargs):
        super().__init__(output_shape_h, *args, **kwargs)
        self.output_shape_c = output_shape_c

    @staticmethod
    def forward(x, h, c, weight_x, weight_h, bias_x, bias_h):
        '''
        Shape:
            - x: [N, input_size]
            - h: [N, hidden_size]
            - c: [N, hidden_size]
            - weight_x: [input_size, 4*hidden_size]
            - weight_h: [hidden_size, 4*hidden_size]
            - bias_x: [4*hidden_size]
            - bias_h: [4*hidden_size]
            - Output_h: [N, hidden_size]
            - Output_c: [N, hidden_size]
        '''
        hidden_size = h.shape[1]
        if bias_x is None or bias_h is None:
            tmp = np.add(np.dot(h.data, weight_h.data), np.dot(x.data, weight_x.data))
        else:
            tmp = np.add(np.add(np.dot(h.data, weight_h.data), bias_x.data), np.add(np.dot(x.data, weight_x.data), bias_h.data))
        f = np.divide(1, np.add(1, np.exp(np.negative(tmp[:,:hidden_size]))))
        g = np.tanh(tmp[:, hidden_size:2*hidden_size])
        i = np.divide(1, np.add(1, np.exp(np.negative(tmp[:,2*hidden_size:3*hidden_size]))))
        o = np.divide(1, np.add(1, np.exp(np.negative(tmp[:,3*hidden_size:4*hidden_size]))))
        c_next = np.add(np.multiply(f, c.data), np.multiply(g, i))
        h_next = np.multiply(o, np.tanh(c_next))
        c_next = Tensor(c_next)
        h_next = Tensor(h_next)
        if bias_x is None or bias_h is None:
            parent = LSTMCell.prepare(h_next.shape, c_next.shape, x, h, c, weight_x, weight_h, bias=False, f=f, g=g, i=i, o=o, c_next=c_next.data, hidden_size=hidden_size)
        else:
            parent = LSTMCell.prepare(h_next.shape, c_next.shape, x, h, c, weight_x, weight_h, bias_x, bias_h, bias=True, f=f, g=g, i=i, o=o, c_next=c_next.data, hidden_size=hidden_size)
        c_next.set_creator(parent)
        h_next.set_creator(parent)
        return h_next, c_next

    def calc_grad(self, dh_next, dc_next):
        tanh_c_next = np.tanh(self.kwargs['c_next'])
        ds = dc_next + (dh_next * self.kwargs['o']) * (1 - tanh_c_next ** 2)
        
        df = ds * self.var[2].data
        dg = ds * self.kwargs['i']
        di = ds * self.kwargs['g']
        do = dh_next * tanh_c_next
        # derivative of activation
        df *= self.kwargs['f'] * (1 - self.kwargs['f'])
        dg *= (1 - self.kwargs['g'] ** 2)
        di *= self.kwargs['i'] * (1 - self.kwargs['i'])
        do *= self.kwargs['o'] * (1 - self.kwargs['o'])
        dtmp = np.hstack((df, dg, di, do))

        dx = np.dot(dtmp, self.var[3].data.T)
        dh = np.dot(dtmp, self.var[4].data.T)
        dc = ds * self.kwargs['f']
        dw_h = np.dot(self.var[1].data.T, dtmp)
        dw_x = np.dot(self.var[0].data.T, dtmp)

        if not self.kwargs['bias']:        
            return dx, dh, dc, dw_x, dw_h
        else:
            hidden_size = self.kwargs['hidden_size']
            db_x = np.zeros_like(self.var[5].data)
            db_h = np.zeros_like(self.var[6].data)
            db_x = LSTMCell.handle_broadcast(dtmp, self.var[5])
            db_h = LSTMCell.handle_broadcast(dtmp, self.var[6])
            return dx, dh, dc, dw_x, dw_h, db_x, db_h

    def backward(self, dh_next, dc_next=None):
        if dc_next is None:
            dc_next = np.zeros_like(self.kwargs['c_next'])
        grads = self.calc_grad(dh_next, dc_next)
        if type(grads) is list:
            grads = tuple(grads)
        if type(grads) is not tuple:
            grads = (grads,)
        for dx, var in zip(grads, self.var):
            if not var.requires_grad:
                continue
            if var.grad is None:
                var.grad = dx
            else:
                var.grad += dx
        for i, var in enumerate(self.var):
            if i == 2:
                continue
            if var.creator is not None:
                if i == 1:
                    if isinstance(var.creator, LSTMCell):
                        var.backward(var.grad, self.var[2].grad)
                    else:
                        var.backward(var.grad)
                else:
                    var.backward(var.grad)

lstmcell = LSTMCell(None, None)

class LSTM(Function):
    @staticmethod
    def forward(x, h, c, weight_x, weight_h, bias_x, bias_h, num_layers):
        '''
        Shape:
            - x: [seq_len, N, input_size]
            - h: [num_layers, N, hidden_size]
            - c: [num_layers, N, hidden_size]
            - Output: [seq_len, N, hidden_size]
            - Hidden_h: [num_layers, N, hidden_size]
            - Hidden_c: [num_layers, N, hidden_size]
        '''
        seq_len = x.shape[0]
        h_out = []
        c_out = []
        tmp = [x]
        for l in range(num_layers):
            hx = h[l]
            cx = c[l]
            tmp.append([])
            for i in range(seq_len):
                if bias_x is None or bias_h is None:
                    hx, cx = lstmcell(tmp[l][i], hx, cx, weight_x[l], weight_h[l], None, None)
                    tmp[l+1].append(hx)
                else:
                    hx, cx = lstmcell(tmp[l][i], hx, cx, weight_x[l], weight_h[l], bias_x[l], bias_h[l])
                    tmp[l+1].append(hx)
            h_out.append(hx)
            c_out.append(cx)
        result_x = listconcat(tmp[-1])
        result_h = listconcat(h_out)
        result_c = listconcat(c_out)
        return result_x, result_h, result_c
    
lstm = LSTM(None)

class GRUCell(Function):
    @staticmethod
    def forward(x, h, weight_x, weight_h, bias_x, bias_h):
        '''
        Shape:
            - x: [N, input_size]
            - h: [N, hidden_size]
            - weight_x: [input_size, 3*hidden_size]
            - weight_h: [hidden_size, 3*hidden_size]
            - bias_x: [3*hidden_size]
            - bias_h: [3*hidden_size]
            - Output: [N, hidden_size]
        '''
        hidden_size = h.shape[1]
        if bias_x is None or bias_h is None:
            tmp = np.add(np.dot(h.data, weight_h.data[:, :2*hidden_size]), np.dot(x.data, weight_x.data[:, :2*hidden_size]))
        else:
            tmp = np.add(np.add(np.dot(h.data, weight_h.data[:, :2*hidden_size]), bias_x.data[:2*hidden_size]), np.add(np.dot(x.data, weight_x.data[:, :2*hidden_size]), bias_h.data[:2*hidden_size]))
        tmp = np.divide(1, np.add(1, np.exp(np.negative(tmp))))
        r = tmp[:, :hidden_size]
        z = tmp[:, hidden_size:]
        if bias_x is None or bias_h is None:
            n = np.tanh(np.add(np.multiply(r, np.dot(h.data, weight_h.data[:, 2*hidden_size:])), np.dot(x.data, weight_x.data[:, 2*hidden_size:])))            
            h_next = Tensor(np.add(np.multiply(np.subtract(1, z), n), np.multiply(z, h.data)))
            h_next.set_creator(GRUCell.prepare(h_next.shape, x, h, weight_x, weight_h, bias=False, r=r, z=z, n=n, tmp=tmp, hidden_size=hidden_size))
        else:
            n = np.tanh(np.add(np.add(np.multiply(r, np.dot(h.data, weight_h.data[:, 2*hidden_size:])), bias_x.data[2*hidden_size:]), np.add(np.dot(x.data, weight_x.data[:, 2*hidden_size:]), bias_h.data[2*hidden_size:])))
            h_next = Tensor(np.add(np.multiply(np.subtract(1, z), n), np.multiply(z, h.data)))
            h_next.set_creator(GRUCell.prepare(h_next.shape, x, h, weight_x, weight_h, bias_x, bias_h, bias=True, r=r, z=z, n=n, tmp=tmp, hidden_size=hidden_size))
        return h_next

    def calc_grad(self, dh_next):
        hidden_size = self.kwargs['hidden_size']
        dw_x = np.zeros_like(self.var[2].data)
        dw_h = np.zeros_like(self.var[3].data)
        dn = np.multiply(dh_next, np.subtract(1, self.kwargs['z']))
        dh = np.multiply(dh_next, self.kwargs['z'])
        # tanh derivative
        tmp = np.multiply(dn, np.subtract(1, np.square(self.kwargs['n'])))
        dx = np.dot(tmp, self.var[2].data[:, 2*hidden_size:].T)
        dw_x[:, 2*hidden_size:] = np.dot(self.var[0].data.T, tmp)
        rtmp = np.multiply(self.kwargs['r'], tmp)
        dh += np.dot(rtmp, self.var[3].data[:, 2*hidden_size:].T)
        dw_h[:, 2*hidden_size:] = np.dot(self.var[1].data.T, rtmp)
        dr = np.multiply(tmp, np.dot(self.var[1].data, self.var[3].data[:, 2*hidden_size:]))
        dz = np.subtract(np.multiply(self.var[1].data, dh_next), np.multiply(self.kwargs['n'], dh_next))   
        dtmp = np.concatenate([dr,dz],axis=1) 
        # sigmoid derivative
        tmp2 = np.multiply(dtmp, np.multiply(self.kwargs['tmp'], np.subtract(1, self.kwargs['tmp'])))
        dx += np.dot(tmp2, self.var[2].data[:, :2*hidden_size].T)
        dw_x[:, :2*hidden_size] = np.dot(self.var[0].data.T, tmp2)
        dh += np.dot(tmp2, self.var[3].data[:, :2*hidden_size].T)
        dw_h[:, :2*hidden_size] = np.dot(self.var[1].data.T, tmp2)
        
        if not self.kwargs['bias']:        
            return dx, dh, dw_x, dw_h
        else:
            db_x = np.zeros_like(self.var[4].data)
            db_h = np.zeros_like(self.var[5].data)
            db_x[2*hidden_size:] = GRUCell.handle_broadcast(tmp, self.var[4][2*hidden_size:])
            db_h[2*hidden_size:] = GRUCell.handle_broadcast(rtmp, self.var[5][2*hidden_size:])
            db_x[:2*hidden_size] = GRUCell.handle_broadcast(tmp2, self.var[4][:2*hidden_size])
            db_h[:2*hidden_size] = GRUCell.handle_broadcast(tmp2, self.var[5][:2*hidden_size])
            return dx, dh, dw_x, dw_h, db_x, db_h

grucell = GRUCell(None)

class GRU(Function):
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
                    hx = grucell(tmp[l][i], hx, weight_x[l], weight_h[l], None, None)
                    tmp[l+1].append(hx)
                else:
                    hx = grucell(tmp[l][i], hx, weight_x[l], weight_h[l], bias_x[l], bias_h[l])
                    tmp[l+1].append(hx)
            h_out.append(hx)
        result_x = listconcat(tmp[-1])
        result_h = listconcat(h_out)
        return result_x, result_h

gru = GRU(None)
