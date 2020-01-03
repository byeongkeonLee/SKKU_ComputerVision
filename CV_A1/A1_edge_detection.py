import A1_image_filtering
import cv2
import numpy as np
import time

IMAGE_FILE_NAME_LIST = ["lenna.png", "shapes.png"]

def compute_image_gradient(img):
    filtered_img = A1_image_filtering.cross_correlation_2d(img,A1_image_filtering.get_gaussian_filter_2d(7,1.5))

    sovel_xx = [[-1,0,1]]
    sovel_xy = [[1], [2], [1]]
    sovel_yx = [[1, 2, 1]]
    sovel_yy = [[-1], [0], [1]]

    sovel_x_img=A1_image_filtering.cross_correlation_1d(filtered_img, sovel_xy)
    sovel_x_img=A1_image_filtering.cross_correlation_1d(sovel_x_img, sovel_xx)

    sovel_y_img = A1_image_filtering.cross_correlation_1d(filtered_img, sovel_yy)
    sovel_y_img = A1_image_filtering.cross_correlation_1d(sovel_y_img, sovel_yx)
    dir = np.arctan2(sovel_y_img,sovel_x_img)
    mag = np.array(np.sqrt(np.square(sovel_x_img)+np.square(sovel_y_img)))
    #print(np.sum(np.sqrt(np.square(sovel_x_img) + np.square(sovel_y_img)) >= 256))
    return mag,dir


def non_maximum_suppression_dir (mag,dir):
    mag_boundary = mag.shape
    dir_forward = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])
    dir_backward = np.array([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]])

    dir_forward = np.array([[0, 1], [-1, -1], [-1, 0], [-1, 1], [0, -1], [1, 1], [1, 0], [1, -1]])
    dir_backward = np.array([[0, -1], [1, 1], [1, 0], [1, -1], [0, 1], [-1, -1], [-1, 0], [-1, 1]])

    dir=(dir*180/np.pi)%360
    dir=dir/45
    dir[dir>=7.5] = 0
    dir=np.around(dir).astype('int')
    indices = np.array(list(np.ndindex((dir.shape[0], dir.shape[1])))).reshape(dir.shape[0], dir.shape[1], 2)
    forward = dir_forward[dir[:]]+indices
    backward = dir_backward[dir[:]]+indices

    output=mag.copy()
    forward[forward[:, :, 0] < 0, 0]=0
    forward[forward[:, :, 1] < 0, 1] = 0
    backward[backward[:, :, 0] < 0, 0] = 0
    backward[backward[:, :, 1] < 0, 1] = 0

#   print(forward[:, :, 0] > mag_boundary[0]-1)
    forward[forward[:, :, 0] > mag_boundary[0]-1,0] = mag_boundary[0]-1
    forward[forward[:, :, 1] > mag_boundary[1]-1,1] = mag_boundary[1]-1

#   print(mag_boundary, np.max(forward[:,:,0]),np.max(forward[:,:,1]))

    backward[backward[:, :, 0] > mag_boundary[0]-1,0] = mag_boundary[0]-1
    backward[backward[:, :, 1] > mag_boundary[1]-1,1] = mag_boundary[1]-1
    output[mag[forward[:, :, 0], forward[:, :, 1]] > mag[:]] = 0
    output[mag[backward[:, :, 0], backward[:, :, 1]] > mag[:]] = 0
    return output

if __name__ =="__main__":

    #IMAGE_FILE_NAME_LIST = ["byeongkeon.png"]

    for IMAGE_NAME in IMAGE_FILE_NAME_LIST:
        IMAGE_FILE_PATH = "image/"+IMAGE_NAME

        img = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_GRAYSCALE)

        start = time.time()
        mag, dir = compute_image_gradient(img)
        print(IMAGE_NAME,"compute_image_gradient Computation time : ", time.time() - start)
        cv2.imshow("part_2_edge_raw_"+IMAGE_NAME, mag / 255)
        cv2.imwrite("./result/part_2_edge_raw_"+IMAGE_NAME, mag)

        start = time.time()
        suppressed_mag = non_maximum_suppression_dir(mag, dir)
        print(IMAGE_NAME,"non_maximum_suppression_dir Computation time : ", time.time() - start)
        cv2.imshow("part_2_edge_sup_"+IMAGE_NAME, suppressed_mag / 255)
        cv2.imwrite("./result/part_2_edge_sup_"+IMAGE_NAME, suppressed_mag)
        cv2.waitKey()
        cv2.destroyAllWindows()
