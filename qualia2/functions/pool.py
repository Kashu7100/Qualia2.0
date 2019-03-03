# -*- coding: utf-8 -*- 
from ..core import *
from ..autograd import *

class MaxPool1d(Function):
    @staticmethod
    def forward(x, kernel_width=2, stride=2, padding=0, dilation=1, return_indices=False):
        '''Applies a 1D max pooling over an input signal composed of several input planes.\n
        Args:
            kernel_width (int):
            stride (int):
            padding (int):
            dilation (int):
            return_indices (bool): if True, will return the max indices along with the outputs.

        Shape:
            - Input: [N,C,W]
            - Output: [N,C,W_out]

            W_out = (W+2*padding-dilation*(kernel_width-1)-1)/stride + 1
        '''
        batch, channel, width = x.shape
 
        ow = int((width+2*padding-dilation*(kernel_width-1)-1)/stride+1)

        padded = np.zeros((batch, channel, width+2*padding))
        padded[:,:,padding:width+padding] = x.data
        reshaped = MaxPool1d._reshape_img(padded, batch, ow, channel, kernel_width, stride, dilation)
        tmp, idx = map(lambda f: np.reshape(f(reshaped, axis=3),(batch, channel, ow)), [np.max, np.argmax])
        result = Tensor(tmp)
        result.set_creator(MaxPool1d.prepare(result.shape, x, idx=idx, kernel_width=kernel_width, padded_shape=padded.shape, stride=stride, dilation=dilation))
        if return_indices:
            return result, idx
        else:
            return result

    @staticmethod
    def _reshape_img(x, batch, ow, channel, kernel_width, stride, dilation):
        fw = (kernel_width-1)*dilation+1
        result = np.zeros((batch, channel, ow, kernel_width))
        for j in range(ow): 
            tmp = x[:, :, j*stride:j*stride+fw] 
            result[:, :, j, :] = tmp[:, :, ::dilation] 
        return result

    @staticmethod
    def _rev_img(delta, kernel_width, argmax, ow, x_shape, padded_shape, stride, dilation): 
        batch, channel, width = x_shape
        _, _, pw = padded_shape 
        fw = (kernel_width-1)*dilation+1
        result = np.zeros(padded_shape)
        for j in range(ow): 
            tmp = np.zeros((batch, channel, fw))
            tmp[:, :, ::dilation][:,:,argmax[:,:,j]] = delta[:,:,j] 
            result[:, :, j*stride:j*stride+fw] += tmp 
        return result[:,:,int((pw-width)/2):pw-int((pw-width)/2)]

    def calc_grad(self, dx):
        _, _, ow = dx.shape
        return MaxPool1d._rev_img(dx, self.kwargs['kernel_width'], self.kwargs['idx'], ow, self.var[0].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])

maxpool1d = MaxPool1d(None)

class MaxPool2d(Function):
    @staticmethod
    def forward(x, kernel_size=(2,2), stride=(2,2), padding=(0,0), dilation=(1,1), return_indices=False): 
        '''Applies a 2D max pooling over an input signal composed of several input planes.\n
        Args:
            kernel_size (tuple of int):
            stride (tuple of int):
            padding (tuple of int):
            dilation (tuple of int):
            return_indices (bool): if True, will return the max indices along with the outputs.

        Shape:
            - Input: [N,C,H,W]
            - Output: [N,C,H_out,W_out]

            H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
            W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
        '''
        batch, channel, height, width = x.shape
        kernel_height, kernel_width = kernel_size

        oh = int((height-2*padding[0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
        ow = int((width+2*padding[1]-dilation[1]*(kernel_width-1)-1)/stride[1]+1)

        padded = np.zeros((batch, channel, height+2*padding[0], width+2*padding[1]))
        padded[:,:,padding[0]:height+padding[0],padding[1]:width+padding[1]] = x.data
        reshaped = MaxPool2d._reshape_img(padded, batch, oh, ow, channel, kernel_height, kernel_width, stride, dilation)
        tmp, idx = map(lambda f: np.reshape(f(reshaped, axis=3),(batch, channel, oh, ow)), [np.max, np.argmax])
        result = Tensor(tmp)
        result.set_creator(MaxPool2d.prepare(result.shape, x, idx=idx, kernel_size=kernel_size, padded_shape=padded.shape, stride=stride, dilation=dilation))
        if return_indices:
            return result, idx
        else:
            return result

    @staticmethod
    def _reshape_img(x, batch, oh, ow, channel, kernel_height, kernel_width, stride, dilation):
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1) 
        result = np.zeros((batch, channel, oh*ow, kernel_height, kernel_width))
        for i in range(oh): 
            for j in range(ow): 
                tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] 
                result[:, :, i*ow+j, :, :] = tmp[:, :, ::dilation[0], ::dilation[1]] 
        return np.reshape(result, (batch, channel, oh*ow, kernel_height*kernel_width))

    @staticmethod
    def _rev_img(delta, kernel_size, argmax, oh, ow, x_shape, padded_shape, stride, dilation): 
        batch, channel, height, width = x_shape 
        kernel_height, kernel_width = kernel_size
        _, _, ph, pw = padded_shape 
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1)
        result = np.zeros(padded_shape)
        for i in range(oh): 
            for j in range(ow): 
                tmp = np.zeros((batch, channel, fh, fw))
                tmp[:, :, ::dilation[0], ::dilation[1]].reshape(batch, channel,-1)[:,:,argmax[:,:,i,j]] = delta[:,:,i,j] 
                result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] += tmp 
        return result[:,:,int((ph-height)/2):ph-int((ph-height)/2),int((pw-width)/2):pw-int((pw-width)/2)]

    def calc_grad(self, dx):
        _, _, oh, ow = dx.shape
        return MaxPool2d._rev_img(dx, self.kwargs['kernel_size'], self.kwargs['idx'], oh, ow, self.var[0].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])

maxpool2d = MaxPool2d(None)

class MaxPool3d(Function):
    @staticmethod
    def forward(x, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), dilation=(1,1,1), return_indices=False): 
        '''Applies a 3D max pooling over an input signal composed of several input planes.\n
        Args:
            kernel_size (tuple of int):
            stride (tuple of int):
            padding (tuple of int):
            dilation (tuple of int):
            return_indices (bool): if True, will return the max indices along with the outputs.

        Shape:
            - Input: [N,C,H,W,D]
            - Output: [N,C,H_out,W_out,D_out]

            H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
            W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
            D_out = (D+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2] + 1
        '''
        batch, channel, height, width, depth = x.shape
        kernel_height, kernel_width, kernel_depth = kernel_size

        oh = int((height-2*padding[0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
        ow = int((width+2*padding[1]-dilation[1]*(kernel_width-1)-1)/stride[1]+1)
        od = int((depth+2*padding[2]-dilation[2]*(kernel_depth-1)-1)/stride[2]+1)

        padded = np.zeros((batch, channel, height+2*padding[0], width+2*padding[1], depth+2*padding[2]))
        padded[:,:,padding[0]:height+padding[0],padding[1]:width+padding[1],padding[2]:depth+padding[2]] = x.data
        reshaped = MaxPool3d._reshape_img(padded, batch, oh, ow, od, channel, kernel_height, kernel_width, kernel_depth, stride, dilation)
        tmp, idx = map(lambda f: np.reshape(f(reshaped, axis=3),(batch, channel, oh, ow, od)), [np.max, np.argmax])
        result = Tensor(tmp)
        result.set_creator(MaxPool3d.prepare(result.shape, x, idx=idx, kernel_size=kernel_size, padded_shape=padded.shape, stride=stride, dilation=dilation))
        if return_indices:
            return result, idx
        else:
            return result

    @staticmethod
    def _reshape_img(x, batch, oh, ow, od, channel, kernel_height, kernel_width, kernel_depth, stride, dilation):
        fh, fw, fd = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1, (kernel_depth-1)*dilation[2]+1) 
        result = np.zeros((batch, channel, oh*ow*od, kernel_height, kernel_width, kernel_depth))
        for i in range(oh): 
            for j in range(ow):
                for k in range(od):
                    tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw, k*stride[2]:k*stride[2]+fd] 
                    result[:, :, i*ow*od+j*ow+k, :, :, :] = tmp[:, :, ::dilation[0], ::dilation[1], ::dilation[2]] 

        return np.reshape(result, (batch, channel, oh*ow*od, kernel_height*kernel_width*kernel_depth))

    @staticmethod
    def _rev_img(delta, kernel_size, argmax, oh, ow, od, x_shape, padded_shape, stride, dilation): 
        batch, channel, height, width, depth = x_shape 
        kernel_height, kernel_width, kernel_depth = kernel_size
        _, _, ph, pw, pd = padded_shape 
        fh, fw, fd = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1, (kernel_depth-1)*dilation[2]+1) 
        result = np.zeros(padded_shape)
        for i in range(oh): 
            for j in range(ow): 
                for k in range(od):
                    tmp = np.zeros((batch, channel, fh, fw, fd))
                    tmp[:, :, ::dilation[0], ::dilation[1], ::dilation[2]].reshape(batch, channel,-1)[:,:,argmax[:,:,i,j,k]] = delta[:,:,i,j,k] 
                    result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw, k*stride[2]:k*stride[2]+fd] += tmp 
        return result[:,:,int((ph-height)/2):ph-int((ph-height)/2),int((pw-width)/2):pw-int((pw-width)/2),int((pd-depth)/2):pd-int((pd-depth)/2)]

    def calc_grad(self, dx):
        _, _, oh, ow, od = dx.shape
        return MaxPool3d._rev_img(dx, self.kwargs['kernel_size'], self.kwargs['idx'], oh, ow, od, self.var[0].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])

maxpool3d = MaxPool3d(None)

class AvePool1d(Function):
    @staticmethod
    def forward(x, kernel_width=2, stride=2, padding=0, dilation=1):
        '''Applies a 1D average pooling over an input signal composed of several input planes.\n
        Args:
            kernel_width (int):
            stride (int):
            padding (int):
            dilation (int):

        Shape:
            - Input: [N,C,W]
            - Output: [N,C,W_out]

            W_out = (W+2*padding-dilation*(kernel_width-1)-1)/stride + 1
        '''
        batch, channel, width = x.shape
 
        ow = int((width+2*padding-dilation*(kernel_width-1)-1)/stride+1)

        padded = np.zeros((batch, channel, width+2*padding))
        padded[:,:,padding:width+padding] = x.data
        reshaped = AvePool1d._reshape_img(padded, batch, ow, channel, kernel_width, stride, dilation)
        result = Tensor(np.reshape(np.average(reshaped, axis=3),(batch, channel, ow)))
        result.set_creator(AvePool1d.prepare(result.shape, x, kernel_width=kernel_width, padded_shape=padded.shape, stride=stride, dilation=dilation))
        return result

    @staticmethod
    def _reshape_img(x, batch, ow, channel, kernel_width, stride, dilation):
        fw = (kernel_width-1)*dilation+1
        result = np.zeros((batch, channel, ow, kernel_width))
        for j in range(ow): 
            tmp = x[:, :, j*stride:j*stride+fw] 
            result[:, :, j, :] = tmp[:, :, ::dilation] 
        return result

    @staticmethod
    def _rev_img(delta, kernel_width, ow, x_shape, padded_shape, stride, dilation): 
        batch, channel, width = x_shape
        _, _, pw = padded_shape 
        fw = (kernel_width-1)*dilation+1
        result = np.zeros(padded_shape)
        for j in range(ow): 
            tmp = np.zeros((batch, channel, fw))
            tmp[:, :, ::dilation] = delta[:,:,j]/kernel_width
            result[:, :, j*stride:j*stride+fw] += tmp 
        return result[:,:,int((pw-width)/2):pw-int((pw-width)/2)]

    def calc_grad(self, dx):
        _, _, ow = dx.shape
        return AvePool1d._rev_img(dx, self.kwargs['kernel_width'], ow, self.var[0].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])

avepool1d = AvePool1d(None)

class AvePool2d(Function):
    @staticmethod
    def forward(x, kernel_size=(2,2), stride=(2,2), padding=(0,0), dilation=(1,1)): 
        '''Applies a 2D average pooling over an input signal composed of several input planes.\n
        Args:
            kernel_size (tuple of int):
            stride (tuple of int):
            padding (tuple of int):
            dilation (tuple of int):

        Shape:
            - Input: [N,C,H,W]
            - Output: [N,C,H_out,W_out]

            H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
            W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
        '''
        batch, channel, height, width = x.shape
        kernel_height, kernel_width = kernel_size

        oh = int((height-2*padding[0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
        ow = int((width+2*padding[1]-dilation[1]*(kernel_width-1)-1)/stride[1]+1)

        padded = np.zeros((batch, channel, height+2*padding[0], width+2*padding[1]))
        padded[:,:,padding[0]:height+padding[0],padding[1]:width+padding[1]] = x.data
        reshaped = AvePool2d._reshape_img(padded, batch, oh, ow, channel, kernel_height, kernel_width, stride, dilation)
        result = Tensor(np.reshape(np.average(reshaped, axis=3),(batch, channel, oh, ow)))
        result.set_creator(AvePool2d.prepare(result.shape, x, kernel_size=kernel_size, padded_shape=padded.shape, stride=stride, dilation=dilation))
        return result

    @staticmethod
    def _reshape_img(x, batch, oh, ow, channel, kernel_height, kernel_width, stride, dilation):
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1) 
        result = np.zeros((batch, channel, oh*ow, kernel_height, kernel_width))
        for i in range(oh): 
            for j in range(ow): 
                tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] 
                result[:, :, i*ow+j, :, :] = tmp[:, :, ::dilation[0], ::dilation[1]] 
        return np.reshape(result, (batch, channel, oh*ow, kernel_height*kernel_width))

    @staticmethod
    def _rev_img(delta, kernel_size, oh, ow, x_shape, padded_shape, stride, dilation): 
        batch, channel, height, width = x_shape 
        kernel_height, kernel_width = kernel_size
        _, _, ph, pw = padded_shape 
        fh, fw = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1)
        result = np.zeros(padded_shape)
        for i in range(oh): 
            for j in range(ow): 
                tmp = np.zeros((batch, channel, fh, fw))
                tmp[:, :, ::dilation[0], ::dilation[1]] = delta[:,:,i,j]/(kernel_height*kernel_width) 
                result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw] += tmp 
        return result[:,:,int((ph-height)/2):ph-int((ph-height)/2),int((pw-width)/2):pw-int((pw-width)/2)]

    def calc_grad(self, dx):
        _, _, oh, ow = dx.shape
        return AvePool2d._rev_img(dx, self.kwargs['kernel_size'], oh, ow, self.var[0].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])

avepool2d = AvePool2d(None)

class AvePool3d(Function):
    @staticmethod
    def forward(x, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), dilation=(1,1,1), return_indices=False): 
        '''Applies a 3D max pooling over an input signal composed of several input planes.\n
        Args:
            kernel_size (tuple of int):
            stride (tuple of int):
            padding (tuple of int):
            dilation (tuple of int):
            return_indices (bool): if True, will return the max indices along with the outputs.

        Shape:
            - Input: [N,C,H,W,D]
            - Output: [N,C,H_out,W_out,D_out]

            H_out = (H+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
            W_out = (W+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1
            D_out = (D+2*padding[2]-dilation[2]*(kernel_size[2]-1)-1)/stride[2] + 1
        '''
        batch, channel, height, width, depth = x.shape
        kernel_height, kernel_width, kernel_depth = kernel_size

        oh = int((height-2*padding[0]-dilation[0]*(kernel_height-1)-1)/stride[0]+1) 
        ow = int((width+2*padding[1]-dilation[1]*(kernel_width-1)-1)/stride[1]+1)
        od = int((depth+2*padding[2]-dilation[2]*(kernel_depth-1)-1)/stride[2]+1)

        padded = np.zeros((batch, channel, height+2*padding[0], width+2*padding[1], depth+2*padding[2]))
        padded[:,:,padding[0]:height+padding[0],padding[1]:width+padding[1],padding[2]:depth+padding[2]] = x.data
        reshaped = AvePool3d._reshape_img(padded, batch, oh, ow, od, channel, kernel_height, kernel_width, kernel_depth, stride, dilation)

        result = Tensor(np.reshape(np.average(reshaped, axis=3),(batch, channel, oh, ow, od)))
        result.set_creator(AvePool3d.prepare(result.shape, x, kernel_size=kernel_size, padded_shape=padded.shape, stride=stride, dilation=dilation))
        return result

    @staticmethod
    def _reshape_img(x, batch, oh, ow, od, channel, kernel_height, kernel_width, kernel_depth, stride, dilation):
        fh, fw, fd = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1, (kernel_depth-1)*dilation[2]+1) 
        result = np.zeros((batch, channel, oh*ow*od, kernel_height, kernel_width, kernel_depth))
        for i in range(oh): 
            for j in range(ow):
                for k in range(od):
                    tmp = x[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw, k*stride[2]:k*stride[2]+fd] 
                    result[:, :, i*ow*od+j*ow+k, :, :, :] = tmp[:, :, ::dilation[0], ::dilation[1], ::dilation[2]] 

        return np.reshape(result, (batch, channel, oh*ow*od, kernel_height*kernel_width*kernel_depth))

    @staticmethod
    def _rev_img(delta, kernel_size, oh, ow, od, x_shape, padded_shape, stride, dilation): 
        batch, channel, height, width, depth = x_shape 
        kernel_height, kernel_width, kernel_depth = kernel_size
        _, _, ph, pw, pd = padded_shape 
        fh, fw, fd = ((kernel_height-1)*dilation[0]+1, (kernel_width-1)*dilation[1]+1, (kernel_depth-1)*dilation[2]+1) 
        result = np.zeros(padded_shape)
        for i in range(oh): 
            for j in range(ow): 
                for k in range(od):
                    tmp = np.zeros((batch, channel, fh, fw, fd))
                    tmp[:, :, ::dilation[0], ::dilation[1], ::dilation[2]] = delta[:,:,i,j,k]/(kernel_height*kernel_width*kernel_depth) 
                    result[:, :, i*stride[0]:i*stride[0]+fh, j*stride[1]:j*stride[1]+fw, k*stride[2]:k*stride[2]+fd] += tmp 
        return result[:,:,int((ph-height)/2):ph-int((ph-height)/2),int((pw-width)/2):pw-int((pw-width)/2),int((pd-depth)/2):pd-int((pd-depth)/2)]

    def calc_grad(self, dx):
        _, _, oh, ow, od = dx.shape
        return AvePool3d._rev_img(dx, self.kwargs['kernel_size'], oh, ow, od, self.var[0].shape, self.kwargs['padded_shape'], self.kwargs['stride'], self.kwargs['dilation'])

avepool3d = AvePool3d(None)