import cv2
import numpy as np
import my_function
import time
import os

IMAGE_FILE_NAME_LIST = ["lenna.png","shapes.png"]

def cross_correlation_1d(img, kernel):
    kernel_size = np.shape(kernel)
    out = np.zeros(img.shape)
    i = 0
    if kernel_size[0]==1: #col
        padding = kernel_size[1]//2
        img_padding = my_function.padding(img,(0,padding), 'edge')
        expanded_img = my_function.convolution_stride(img_padding,(1,kernel_size[1]))
        return np.einsum('ijkl,kl->ij',expanded_img,kernel)
    else:#row
        padding = kernel_size[0]//2
        img_padding = my_function.padding(img,(padding,0), 'edge')
        expanded_img = my_function.convolution_stride(img_padding, (kernel_size[0],1))
        return np.einsum('ijkl,kl->ij', expanded_img, kernel)

def cross_correlation_2d(img, kernel):
    padding = np.shape(kernel)[0]//2
    img_padding = my_function.padding(img,(padding,padding),'edge')
    expanded_input = my_function.convolution_stride(img_padding,kernel.shape)
    return np.einsum('ijkl,kl->ij', expanded_input, kernel)

def get_gaussian_filter_1d(size,sigma):
    i = np.linspace(-size//2+1,size//2,size)
    kernel = 1/(2* np.pi * np.square(sigma))*np.exp(-(np.square(i))/(2*np.square(sigma)))
    kernel = kernel/np.sum(kernel)
    return np.reshape(kernel,(1,kernel.shape[0])), np.reshape(kernel,(-1,1))

def get_gaussian_filter_2d(size,sigma):
    i = np.linspace(-size//2+1,size//2,size)
    xx,yy = np.meshgrid(i,i)
    kernel = 1/(2* np.pi * np.square(sigma))*np.exp(-(np.square(xx)+np.square(yy))/(2*np.square(sigma)))
    return np.array(kernel*1/np.sum(kernel))


if __name__ =="__main__":


    #IMAGE_FILE_NAME_LIST=["byeongkeon.png"]
    try:
        if not os.path.exists("./result"):
            os.makedirs("./result")
    except OSError:
        print("ERROR for Creating Directory")

    print("get_gaussian_filter_1d(5,1)\n", get_gaussian_filter_1d(5, 1)[0])
    print("get_gaussian_filter_2d(5,1)\n", get_gaussian_filter_2d(5, 1))

    for IMAGE_NAME in IMAGE_FILE_NAME_LIST:
        IMAGE_FILE_PATH = "image/"+IMAGE_NAME
        img = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_GRAYSCALE)
        img_out = []
        for i in [5, 11, 17]:
            col_img = []
            for j in [1, 6, 11]:
                # print(i,j)
                kernel = get_gaussian_filter_2d(i, j)
                filtered_img = cross_correlation_2d(img, kernel)
                cv2.putText(filtered_img, str(i) + "x" + str(i) + " s=" + str(j), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (0, 0, 0), 1, cv2.LINE_AA)
                if j == 1:
                    col_img = filtered_img
                else:
                    col_img = np.concatenate((col_img, filtered_img), axis=1)
            if i == 5:
                img_out = col_img
            else:
                img_out = np.concatenate((img_out, col_img), axis=0)

        start = time.time()
        gaussian_1d = cross_correlation_1d(img, get_gaussian_filter_1d(5, 1)[0])
        gaussian_1d = cross_correlation_1d(gaussian_1d, get_gaussian_filter_1d(5, 1)[1])
        gaussian_2d = cross_correlation_2d(img,get_gaussian_filter_2d(5, 1))
        cv2.imshow("part_1_gaussian_difference_between_1d_and_2d",gaussian_1d-gaussian_2d)
        print(IMAGE_NAME,"Error : ", np.sum(np.abs(gaussian_1d - gaussian_2d)))
        print(IMAGE_NAME,"sobel1d and sobel2d Computation time : ", time.time() - start)
        cv2.imshow("part_1_gaussian_filtered_"+IMAGE_NAME, img_out / 256)
        cv2.imwrite("./result/part_1_gaussian_filtered_"+IMAGE_NAME, img_out)

        cv2.waitKey()
        cv2.destroyAllWindows()