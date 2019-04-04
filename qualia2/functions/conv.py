# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class Conv1d(Function):
    @staticmethod
    def forward(x, kernel, bias=None, stride=1, padding=1, dilation=1):
        '''Applies a 2D convolution over an input signal composed of several input planes.\n 
        Args: 
            x (Tensor): Input tensor with shepe of [batch, channel, width] 
            kernel (Tensor): Kernel with shape of [patch, channel, kernel_width] 
            bias (Tensor): Bias with shape of [patch] to add if needed. Default: None 
            stride (int): Stride of the convolution. Default: 1
            padding (int): Padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension. Default: 1
            dilation (int): Spacing between kernel elements. Default: 1
        Returns: 
            (Tensor): Output tensor will have shape of [batch, patch, out_width] 

        Shape: 
            - Input: [N, in_channels, W] 
            - Output: [N, out_channels, W_out] 
 
            W_out = (W+2*padding-dilation*(kernel_size-1)-1)/stride+1 
        ''' 
        batch, channel, width = x.shape 
        patch, _, kernel_width = kernel.shape 

        ow = int((width+2*padding-dilation*(kernel_width-1)-1)/stride+1)

        padded = np.zeros((batch, channel, width+2*padding))
        padded[:,:,padding:width+padding] = x.data
        reshaped = Conv1d.unfold(padded, batch, ow, kernel.shape, stride, dilation)
        if bias is None:
            result = Tensor(np.tensordot(reshaped, kernel.data, ((2,3),(1,2))).transpose(0,2,1))
            result.set_creator(Conv1d.prepare(result.shape, x, kernel, bias=False, reshaped=reshaped, stride=stride, padded_shape=padded.shape, dilation=dilation, ow=ow))
        else:
            result = Tensor(np.add(np.tensordot(reshaped, kernel.data, ((2,3),(1,2))).transpose(0,2,1), np.reshape(bias.data, (1,-1,1))))
            result.set_creator(Conv1d.prepare(result.shape, x, kernel, bias, bias=True, reshaped=reshaped, stride=stride, padded_shape=padded.shape, dilation=dilation, ow=ow))
        return result

    @staticmethod
    def unfold(x, batch, ow, kernel_shape, stride, dilation): 
        _, _, xw = x.shape
        _, channel, kernel_width = kernel_shape 
        fw = (kernel_width-1)*dilation+1
        result = np.zeros((batch, ow, channel, kernel_width)) 
        for j in range(ow):
            if j*stride+fw > xw:
                continue
            tmp = x[:, :, j*stride:j*stride+fw] 
            result[:, j, :, :] = tmp[:, :, ::dilation] 
        return result 

    @staticmethod
    def fold(delta, ow, x_shape, kernel_shape, padded_shape, stride, dilation):
        batch, channel, width = x_shape 
        _, _, kernel_width = kernel_shape
        _, _, pw = padded_shape 
        fw = (kernel_width-1)*dilation+1
        result = np.zeros(padded_shape)
        for j in range(ow): 
            if j*stride+fw > pw:
                continue
            tmp = np.zeros((batch, channel, fw))
            tmp[:, :, ::dilation] = delta[:, j, :, :] 
            result[:, :, j*stride:j*stride+fw] += tmp
        return result[:,:,int((pw-width)/2):pw-int((pw-width)/2)]
    
    def calc_grad(self, dx):
        batch, patch, _ = dx.shape 
        delta = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.var[1].data, (1,0))
        delta = Conv1d.fold(delta, self.kwargs['ow'], self.var[0].shape, self.var[1].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])
        dk = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.kwargs['reshaped'], ((0,2),(0,1))) 
        if not self.kwargs['bias']:
            return delta, dk
        else:
            db = Conv1d.handle_broadcast(dx, self.var[2])
            return delta, dk, db

conv1d = Conv1d(None)

class Conv2d(Function):        
    @staticmethod
    def forward(x, kernel, bias=None, stride=(1,1), padding=(1,1), dilation=(1,1)):
        '''Applies a 2D convolution over an input signal composed of several input planes.\n 
        Args: 
            x (Tensor): Input tensor with shepe of [batch, channel, height, width] 
            kernel (Tensor): Kernel with shape of [patch, channel, kernel_height, kernel_width] 
            bias (Tensor): Bias with shape of [patch] to add if needed. Default: None 
            stride (tuple of int): Stride of the convolution. Default: (1,1) 
            padding (tuple of int): Padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension. Default: (1,1)
            dilation (tuple of int): Spacing between kernel elements. Default: (1,1)
        Returns: 
            (Tensor): Output tensor will have shape of [batch, patch, out_height, out_width] 

        Shape: 
            - Input: [N, in_channels, H, W] 
            - Output: [N, out_channels, H_out, W_out] 
 
            H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1 
            W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1 
        ''' 
        batch, channel, height, width = x.shape 
        patch, _, kernel_height, kernel_width = kernel.shape 

        oh = int((height+2*padding[0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
        ow = int((width+2*padding[1]-dilation[1]*(kernel_width-1)-1)/stride[1]+1)

        padded = np.zeros((batch, channel, height+2*padding[0], width+2*padding[1]))
        padded[:,:,padding[0]:height+padding[0],padding[1]:width+padding[1]] = x.data
        reshaped = Conv2d.unfold(padded, batch, oh, ow, kernel.shape, stride, dilation)
        if bias is None: 
            result = Tensor(np.tensordot(reshaped, kernel.data, ((2,3,4),(1,2,3))).transpose(0,2,1).reshape(-1,patch,oh,ow)) 
            result.set_creator(Conv2d.prepare(result.shape, x, kernel, bias=False, oh=oh, ow=ow, reshaped=reshaped, padded_shape=padded.shape, stride=stride, dilation=dilation)) 
        else: 
            result = Tensor(np.add(np.tensordot(reshaped, kernel.data, ((2,3,4),(1,2,3))).transpose(0,2,1).reshape(-1,patch,oh,ow), np.reshape(bias.data, (1,-1,1,1)))) 
            result.set_creator(Conv2d.prepare(result.shape, x, kernel, bias, bias=True, oh=oh, ow=ow, reshaped=reshaped, padded_shape=padded.shape, stride=stride, dilation=dilation))
        return result 
    
    @staticmethod
    def unfold(x, batch, oh, ow, kernel_shape, stride, dilation):
        _, _, xh, xw = x.shape 
        _, channel, kernel_height, kernel_width = kernel_shape 
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1) 
        result = np.zeros((batch, oh*ow, channel, kernel_height, kernel_width)) 
        for i in range(oh): 
            for j in range(ow): 
                if i*stride[0]+fh > xh or j*stride[1]+fw > xh:
                    continue
                tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] 
                result[:, i*ow+j, :, :, :] = tmp[:, :, ::dilation[0], ::dilation[1]] 
        return result 

    @staticmethod
    def fold(delta, oh, ow, x_shape, kernel_shape, padded_shape, stride, dilation):
        batch, channel, height, width = x_shape 
        _, _, kernel_height, kernel_width = kernel_shape
        _, _, ph, pw = padded_shape 
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1)
        result = np.zeros(padded_shape)
        for i in range(oh): 
            for j in range(ow): 
                if i*stride[0]+fh > ph or j*stride[1]+fw > pw:
                    continue
                tmp = np.zeros((batch, channel, fh, fw))
                tmp[:, :, ::dilation[0], ::dilation[1]] = delta[:, i*ow+j, :, :, :] 
                result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] += tmp
        return result[:,:,int((ph-height)/2):ph-int((ph-height)/2),int((pw-width)/2):pw-int((pw-width)/2)]

    def calc_grad(self, dx):
        batch, patch, _, _ = dx.shape 
        delta = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.var[1].data, (1,0))
        delta = Conv2d.fold(delta, self.kwargs['oh'], self.kwargs['ow'], self.var[0].shape, self.var[1].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])
        dk = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.kwargs['reshaped'], ((0,2),(0,1))) 
        if not self.kwargs['bias']:
            return delta, dk
        else:
            db = Conv2d.handle_broadcast(dx, self.var[2])
            return delta, dk, db

conv2d = Conv2d(None)

class Conv3d(Function):
    @staticmethod
    def forward(x, kernel, bias=None, stride=(1,1,1), padding=(1,1,1), dilation=(1,1,1)):
        '''Applies a 3D convolution over an input signal composed of several input planes.\n 
        Args: 
            x (Tensor): Input tensor with shepe of [batch, channel, height, width, depth] 
            kernel (Tensor): Kernel with shape of [patch, channel, kernel_height, kernel_width, kernel_depth] 
            bias (Tensor): Bias with shape of [patch] to add if needed. Default: None
            stride (tuple of int): Stride of the convolution. Default: (1,1,1) 
            padding (tuple of int): Padding controls the amount of implicit zero-paddings on both sides for padding number of points for each dimension. Default: (1,1,1)
            dilation (tuple of int): Spacing between kernel elements. Default: (1,1,1)
        Returns: 
            (Tensor): Output tensor will have shape of [batch, patch, out_height, out_width, out_depth] 

        Shape: 
            - Input: [N, in_channels, H, W, D] 
            - Output: [N, out_channels, H_out, W_out, D_out] 
 
            H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1 
            W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1 
            D_out = (D+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2]+1 
        ''' 
        batch, channel, height, width, depth = x.shape 
        patch, _, kernel_height, kernel_width, kernel_depth = kernel.shape 

        oh = int((height+2*padding[0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
        ow = int((width+2*padding[1]-dilation[1]*(kernel_width-1)-1)/stride[1]+1)
        od = int((depth+2*padding[2]-dilation[2]*(kernel_depth-1)-1)/stride[2]+1)

        padded = np.zeros((batch, channel, height+2*padding[0], width+2*padding[1], depth+2*padding[2]))
        padded[:,:,padding[0]:height+padding[0],padding[1]:width+padding[1],padding[2]:depth+padding[2]] = x.data
        reshaped = Conv3d.unfold(padded, batch, oh, ow, od, kernel.shape, stride, dilation)
        if bias is None: 
            result = Tensor(np.tensordot(reshaped, kernel.data, ((2,3,4,5),(1,2,3,4))).transpose(0,2,1).reshape(-1,patch,oh,ow,od)) 
            result.set_creator(Conv3d.prepare(result.shape, x, kernel, bias=False, oh=oh, ow=ow, od=od, reshaped=reshaped, padded_shape=padded.shape, stride=stride, dilation=dilation)) 
        else: 
            result = Tensor(np.add(np.tensordot(reshaped, kernel.data, ((2,3,4,5),(1,2,3,4))).transpose(0,2,1).reshape(-1,patch,oh,ow,od), np.reshape(bias.data, (1,-1,1,1,1)))) 
            result.set_creator(Conv3d.prepare(result.shape, x, kernel, bias, bias=True, oh=oh, ow=ow, od=od, reshaped=reshaped, padded_shape=padded.shape, stride=stride, dilation=dilation))
        return result 
    
@staticmethod
    def unfold(x, batch, oh, ow, od, kernel_shape, stride, dilation): 
        _, _, xh, xw, xd = x.shape 
        _, channel, kernel_height, kernel_width, kernel_depth = kernel_shape 
        fh, fw, fd = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1, (kernel_depth-1)*dilation[2]+1) 
        result = np.zeros((batch, oh*ow*od, channel, kernel_height, kernel_width, kernel_depth)) 
        for i in range(oh): 
            for j in range(ow): 
                for k in range(od):
                    if i*stride[0]+fh > xh or j*stride[1]+fw > xw or k*stride[2]+fd > xd:
                        continue
                    tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw, k*stride[2]:k*stride[2]+fd] 
                    result[:, i*ow*od+j*ow+k, :, :, :, :] = tmp[:, :, ::dilation[0], ::dilation[1], ::dilation[2]] 
        return result 

    @staticmethod
    def fold(delta, oh, ow, od, x_shape, kernel_shape, padded_shape, stride, dilation):
        batch, channel, height, width, depth = x_shape 
        _, _, kernel_height, kernel_width, kernel_depth = kernel_shape
        _, _, ph, pw, pd = padded_shape 
        fh, fw, fd = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1, (kernel_depth-1)*dilation[2]+1) 
        result = np.zeros(padded_shape)
        for i in range(oh): 
            for j in range(ow): 
                for k in range(od):
                    if i*stride[0]+fh > ph or j*stride[1]+fw > pw or k*stride[2]+fd > pd:
                        continue
                    tmp = np.zeros((batch, channel, fh, fw, fd))
                    tmp[:, :, ::dilation[0], ::dilation[1], ::dilation[2]] = delta[:, i*ow*od+j*ow+k, :, :, :, :] 
                    result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw, k*stride[2]:k*stride[2]+fd] += tmp
        return result[:,:,int((ph-height)/2):ph-int((ph-height)/2),int((pw-width)/2):pw-int((pw-width)/2),int((pd-depth)/2):pd-int((pd-depth)/2)]
    
    def calc_grad(self, dx):
        batch, patch, _, _, _ = dx.shape 
        delta = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.var[1].data, (1,0))
        delta = Conv3d.fold(delta, self.kwargs['oh'], self.kwargs['ow'], self.kwargs['od'], self.var[0].shape, self.var[1].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])
        dk = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.kwargs['reshaped'], ((0,2),(0,1))) 
        if not self.kwargs['bias']:
            return delta, dk
        else:
            db = Conv3d.handle_broadcast(dx, self.var[2])
            return delta, dk, db

conv3d = Conv3d(None)

class ConvTranspose1d(Function):
    @staticmethod
    def forward(x, kernel, bias=None, stride=1, padding=1, output_padding=0, dilation=1):
        '''Applies a 1D transposed convolution over an input signal composed of several input planes.\n 
        Args: 
            x (Tensor): Input tensor with shepe of [batch, channel, height, width] 
            kernel (Tensor): Kernel with shape of [patch, channel, kernel_height, kernel_width] 
            bias (Tensor): Bias with shape of [patch] to add if needed. Default: None 
            stride (tuple of int): Stride of the convolution. Default: 1
            padding (tuple of int):  Zero-padding added to both sides of the input. Default: 1
            output_padding (tuple of int): Zero-padding added to both sides of the output. Default: 0
            dilation (tuple of int): Spacing between kernel elements. Default: 1
     
        Shape: 
            - Input: [N, in_channels, H] 
            - Output: [N, out_channels, H_out] 
 
            H_out = (H-1)*stride[0]-2*padding[0]+dilation[0]*(kernel_size[0]-1)+1+output_padding[0]

        Reference:
            https://arxiv.org/pdf/1603.07285.pdf
        ''' 
        batch, channel, height = x.shape 
        patch, _, kernel_height = kernel.shape 

        # output padding term is purposely dropped from oh and oh calculation below.
        oh = int((height-1)*stride-2*padding+dilation*(kernel_height-1)+1)

        offset_h = dilation*(kernel_height-1)-padding
        padded = np.zeros((batch, channel, (height-1)*stride+1+offset_h*2))
        padded[:,:,offset_h:(height-1)*stride+1+offset_h][:,:, ::stride] = x.data
        reshaped = Conv1d.unfold(padded, batch, oh, kernel.shape, stride, dilation)
        if bias is None: 
            tmp = np.tensordot(reshaped, kernel.data, ((2,3),(1,2))).transpose(0,2,1).reshape(-1,patch,oh)
            out = np.zeros((batch, patch, oh+2*output_padding))
            out[:,:,output_padding:oh+output_padding] = tmp
            result = Tensor(out) 
            result.set_creator(ConvTranspose1d.prepare(result.shape, x, kernel, bias=False, oh=oh, reshaped=reshaped, padded_shape=padded.shape, stride=stride, output_padding=output_padding, dilation=dilation)) 
        else:
            tmp = np.add(np.tensordot(reshaped, kernel.data, ((2,3),(1,2))).transpose(0,2,1).reshape(-1,patch,oh), np.reshape(bias.data, (1,-1,1)))
            out = np.zeros((batch, patch, oh+2*output_padding))
            out[:,:,output_padding:oh+output_padding] = tmp
            result = Tensor(out) 
            result.set_creator(ConvTranspose1d.prepare(result.shape, x, kernel, bias, bias=True, oh=oh, reshaped=reshaped, padded_shape=padded.shape, stride=stride, output_padding=output_padding, dilation=dilation))
        return result 

    def calc_grad(self, dx):
        batch, patch, _ = dx.shape 
        dx = dx[:,:,self.kwargs['output_padding']:self.kwargs['oh']+self.kwargs['output_padding']]
        delta = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.var[1].data, (1,0))
        delta = Conv1d.fold(delta, self.kwargs['oh'], self.var[0].shape, self.var[1].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])
        dk = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.kwargs['reshaped'], ((0,2),(0,1))) 
        if not self.kwargs['bias']:
            return delta, dk
        else:
            db = ConvTranspose2d.handle_broadcast(dx, self.var[2])
            return delta, dk, db

convtranspose1d = ConvTranspose1d(None)

class ConvTranspose2d(Function):
    @staticmethod
    def forward(x, kernel, bias=None, stride=(1,1), padding=(1,1), output_padding=(0,0), dilation=(1,1)):
        '''Applies a 2D transposed convolution over an input signal composed of several input planes.\n 
        Args: 
            x (Tensor): Input tensor with shepe of [batch, channel, height, width] 
            kernel (Tensor): Kernel with shape of [patch, channel, kernel_height, kernel_width] 
            bias (Tensor): Bias with shape of [patch] to add if needed. Default: None 
            stride (tuple of int): Stride of the convolution. Default: (1,1)
            padding (tuple of int):  Zero-padding added to both sides of the input. Default: (1,1)
            output_padding (tuple of int): Zero-padding added to both sides of the output. Default: (0,0)
            dilation (tuple of int): Spacing between kernel elements. Default: (1,1)
     
        Shape: 
            - Input: [N, in_channels, H, W] 
            - Output: [N, out_channels, H_out, W_out] 
 
            H_out = (H-1)*stride[0]-2*padding[0]+dilation[0]*(kernel_size[0]-1)+1+output_padding[0]
            W_out = (W-1)*stride[1]-2*padding[1]+dilation[1]*(kernel_size[1]-1)+1+output_padding[1]

        Reference:
            https://arxiv.org/pdf/1603.07285.pdf
        ''' 
        batch, channel, height, width = x.shape 
        patch, _, kernel_height, kernel_width = kernel.shape 

        # output padding term is purposely dropped from oh and oh calculation below.
        oh = int((height-1)*stride[0]-2*padding[0]+dilation[0]*(kernel_height-1)+1) 
        ow = int((width-1)*stride[1]-2*padding[1]+dilation[1]*(kernel_width-1)+1)

        offset_h = dilation[0]*(kernel_height-1-padding[0])
        offset_w = dilation[1]*(kernel_width-1-padding[1])
        padded = np.zeros((batch, channel, (height-1)*stride[0]+1+offset_h*2, (width-1)*stride[1]+1+offset_w*2))
        padded[:,:,offset_h:(height-1)*stride[0]+1+offset_h,offset_w:(width-1)*stride[1]+1+offset_w][:,:, ::stride[0], ::stride[1]] = x.data
        reshaped = Conv2d.unfold(padded, batch, oh, ow, kernel.shape, stride, dilation)
        if bias is None: 
            tmp = np.tensordot(reshaped, kernel.data, ((2,3,4),(1,2,3))).transpose(0,2,1).reshape(-1,patch,oh,ow)
            out = np.zeros((batch, patch, oh+2*output_padding[0], ow+2*output_padding[1]))
            out[:,:,output_padding[0]:oh+output_padding[0],output_padding[1]:ow+output_padding[1]] = tmp
            result = Tensor(out) 
            result.set_creator(ConvTranspose2d.prepare(result.shape, x, kernel, bias=False, oh=oh, ow=ow, reshaped=reshaped, padded_shape=padded.shape, stride=stride, output_padding=output_padding, dilation=dilation)) 
        else:
            tmp = np.add(np.tensordot(reshaped, kernel.data, ((2,3,4),(1,2,3))).transpose(0,2,1).reshape(-1,patch,oh,ow), np.reshape(bias.data, (1,-1,1,1)))
            out = np.zeros((batch, patch, oh+2*output_padding[0], ow+2*output_padding[1]))
            out[:,:,output_padding[0]:oh+output_padding[0],output_padding[1]:ow+output_padding[1]] = tmp
            result = Tensor(out) 
            result.set_creator(ConvTranspose2d.prepare(result.shape, x, kernel, bias, bias=True, oh=oh, ow=ow, reshaped=reshaped, padded_shape=padded.shape, stride=stride, output_padding=output_padding, dilation=dilation))
        return result 

    def calc_grad(self, dx):
        batch, patch, _, _ = dx.shape 
        dx = dx[:,:,self.kwargs['output_padding'][0]:self.kwargs['oh']+self.kwargs['output_padding'][0], self.kwargs['output_padding'][1]:self.kwargs['ow']+self.kwargs['output_padding'][1]]
        delta = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.var[1].data, (1,0))
        delta = Conv2d.fold(delta, self.kwargs['oh'], self.kwargs['ow'], self.var[0].shape, self.var[1].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])
        dk = np.tensordot(np.reshape(dx,(batch,patch,-1)), self.kwargs['reshaped'], ((0,2),(0,1))) 
        if not self.kwargs['bias']:
            return delta, dk
        else:
            db = ConvTranspose2d.handle_broadcast(dx, self.var[2])
            return delta, dk, db

convtranspose2d = ConvTranspose2d(None)
