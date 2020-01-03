import numpy as np
from numpy.lib.stride_tricks import as_strided


def padding(arr,size,mode,constant_values=None):
    out_arr=np.zeros((arr.shape[0]+size[0]*2,arr.shape[1]+size[1]*2))
    #print(out_arr.shape)
    if mode=="constant":
        out_arr.fill(constant_values)
        out_arr[size[0]:size[0]+arr.shape[0],size[1]:size[1]+arr.shape[1]]=arr
        #print(out_arr)
    if mode=="edge":
        out_arr[size[0]:size[0] + arr.shape[0], size[1]:size[1] + arr.shape[1]] = arr
        out_arr[0:size[0],0:size[1]]=arr[0][0]
        out_arr[0:size[0],size[1]:size[1]+arr.shape[1]]=arr[0,:]
        out_arr[0:size[0],size[1]+arr.shape[1]:]=arr[0,arr.shape[1]-1]

        out_arr[size[0]+arr.shape[0]:,0:size[1]] = arr[arr.shape[0]-1,0]
        out_arr[size[0]+arr.shape[0]:, size[1]:size[1]+arr.shape[1]] = arr[arr.shape[0]-1, :]
        out_arr[size[0]+arr.shape[0]:, size[1] + arr.shape[1]:] = arr[arr.shape[0]-1, arr.shape[1] - 1]


        out_arr[size[0]:size[0]+arr.shape[0], 0:size[1]] = np.reshape(arr[:,0],(-1,1))
        out_arr[size[0]:size[0] + arr.shape[0], size[1]+arr.shape[1]:] = np.reshape(arr[:, arr.shape[1]-1],(-1,1))
    return out_arr

#padding(np.array([[1,2,3,4],[5,6,7,8],[10,20,30,40],[11,12,13,14]]),(0,3),mode='edge')


def convolution_stride(arr,kernel_size):
    #print(kernel_size)
    return as_strided(
        arr,
        shape=(
            arr.shape[0] - kernel_size[0] + 1,
            arr.shape[1] - kernel_size[1] + 1,
            kernel_size[0],
            kernel_size[1],
        ),
        strides=(
            arr.strides[0],
            arr.strides[1],
            arr.strides[0],
            arr.strides[1],
        ),
        writeable=False,
    )