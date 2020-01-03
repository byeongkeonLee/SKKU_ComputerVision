import A1_image_filtering
import cv2
import numpy as np
import my_function
import time

IMAGE_FILE_NAME_LIST = ["lenna.png", "shapes.png"]

def compute_corner_response(img):
    A1_image_filtering.get_gaussian_filter_2d(7,1.5)
    filtered_img = A1_image_filtering.cross_correlation_2d(img, A1_image_filtering.get_gaussian_filter_2d(7, 1.5))

    sovel_xx = [[-1, 0, 1]]
    sovel_xy = [[1], [2], [1]]
    sovel_yx = [[1, 2, 1]]
    sovel_yy = [[-1], [0], [1]]

    sovel_x_img = A1_image_filtering.cross_correlation_1d(filtered_img, sovel_xy)
    sovel_x_img = A1_image_filtering.cross_correlation_1d(sovel_x_img, sovel_xx)
    sovel_y_img = A1_image_filtering.cross_correlation_1d(filtered_img, sovel_yy)
    sovel_y_img = A1_image_filtering.cross_correlation_1d(sovel_y_img, sovel_yx)

    ix2 = sovel_x_img*sovel_x_img
    ixy = sovel_x_img*sovel_y_img
    iy2 = sovel_y_img*sovel_y_img

    ix2_padding = my_function.padding(ix2,(2,2),'constant', constant_values=0)
    ixy_padding = my_function.padding(ixy,(2,2),'constant', constant_values=0)
    iy2_padding = my_function.padding(iy2,(2,2),'constant', constant_values=0)

    kernel =np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    ix2_expanded = my_function.convolution_stride(ix2_padding,kernel.shape)
    ixy_expanded = my_function.convolution_stride(ixy_padding, kernel.shape)
    iy2_expanded = my_function.convolution_stride(iy2_padding, kernel.shape)
    Mix2=np.einsum('ijkl,kl->ij',ix2_expanded,kernel)
    Mixy=np.einsum('ijkl,kl->ij', ixy_expanded, kernel)
    Miy2=np.einsum('ijkl,kl->ij', iy2_expanded, kernel)

    #print("Mix2",Mix2)
    #print("Mixy", Mixy)
    #print("Miy2", Miy2)
    M=np.zeros((Mix2.shape[0],Mix2.shape[1],2,2))
    M[:,:,0,0]=Mix2
    M[:, :,0,1] = Mixy
    M[:, :,1,0] = Mixy
    M[:, :,1,1] = Miy2
    R = np.linalg.det(M[:,:]) - 0.04 * np.square(M[:,:,0,0]+M[:,:,1,1])
    #print(np.trace(M[:,:,:,:]))
    #print(M[:,:,0,0]+M[:,:,1,1])
    #print("det,",np.linalg.det(M[:,:]),"trace",np.trace(M[:,:]))
    #print(np.linalg.det(M[:,:]) - 0.04 * np.square(np.trace(M[:,:])))
    R[R<0]=0
    R/=np.max(R)

    ##################################

    threshold = 0.1

    """
    img_copy = img.copy()
    img_copy=cv2.cvtColor(img_copy,cv2.COLOR_GRAY2BGR)
    img_copy[R>threshold] = [0,255,0]
    """
    return R

def non_maximum_suppression_win(R, winSize):
    threshold=0.1
    R_pad = my_function.padding(R,(winSize//2,winSize//2),'constant',constant_values=0)
    expanded_R = my_function.convolution_stride(R_pad,(winSize,winSize))
    R_max_eachpoint = np.max(expanded_R,axis=(2,3))
    R_suppressed=R.copy()
    R_suppressed[R_max_eachpoint>R]=0
    R_suppressed[R_suppressed<threshold] = 0
    return R_suppressed

if __name__=="__main__":


    #IMAGE_FILE_NAME_LIST = ["byeongkeon.png"]

    for IMAGE_NAME in IMAGE_FILE_NAME_LIST:
        IMAGE_FILE_PATH = "image/" + IMAGE_NAME
        img = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_GRAYSCALE)

        start = time.time()
        R = compute_corner_response(img)
        print(IMAGE_NAME,"compute_corner_response Computation time : ", time.time() - start)

        cv2.imshow("part_3_R_"+IMAGE_NAME, R)
        cv2.imwrite("./result/part_3_corner_raw_"+IMAGE_NAME,R*255)

        threshold = 0.1
        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        img_copy[R > threshold] = [0, 255, 0]
        cv2.imshow("part_3_corner_bin_"+IMAGE_NAME, img_copy)
        cv2.imwrite("./result/part_3_corner_bin_"+IMAGE_NAME, img_copy)

        start = time.time()
        suppressed_R = non_maximum_suppression_win(R, 11)
        print(IMAGE_NAME,"non_maximum_suppression_win time : ", time.time() - start)

        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        marked = np.where(suppressed_R > 0.1)
        for i in range(len(marked[1])):
            img_copy = cv2.circle(img_copy, (marked[1][i], marked[0][i]), 4, (0, 255, 0), 2)
        cv2.imshow("part_3_corner_sup_"+IMAGE_NAME, img_copy)

        cv2.imwrite("./result/part_3_corner_sup_"+IMAGE_NAME, img_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()