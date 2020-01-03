import cv2
import numpy as np
import compute_avg_reproj_error
import time

house_file = ["./house1.jpg","./house2.jpg","./house_matches.txt"]
library_file = ["./library1.jpg","./library2.jpg","./library_matches.txt"]
temple_file = ["./temple1.png","./temple2.png","./temple_matches.txt"]
image_list = [temple_file, house_file, library_file]
#image_list = [temple_file]
image_size = np.zeros(4)
def compute_F_raw(M):
    A = np.zeros((M.shape[0],9))
    A[:, 0] = M[:, 0] * M[:, 2]
    A[:, 1] = M[:, 0] * M[:, 3]
    A[:, 2] = M[:, 0]
    A[:, 3] = M[:, 1] * M[:, 2]
    A[:, 4] = M[:, 1] * M[:, 3]
    A[:, 5] = M[:, 1]
    A[:, 6] = M[:, 2]
    A[:, 7] = M[:, 3]
    A[:, 8] = 1
    U, S, V = np.linalg.svd(A)
    return np.reshape(V[np.argmin(S)],(3,-1))

def compute_F_norm(M):
    width = image_size[0]/2
    height = image_size[1]/2
    width2 = image_size[2]/2
    height2 = image_size[3]/2
    meanval = np.array([width, height, width2, height2])
    M_norm = M - meanval
    M_norm = M_norm / np.array([width, height, width2, height2])

    T1 = np.array([[1 / width, 0, 0], [0, 1 / height, 0], [0, 0, 1]]) @ np.array([[1, 0, -meanval[0]], [0, 1, -meanval[1]], [0, 0, 1]])
    T2 = np.array([[1 / width2, 0, 0], [0, 1 / height2, 0], [0, 0, 1]]) @ np.array([[1, 0, -meanval[2]], [0, 1, -meanval[3]], [0, 0, 1]])

    A = np.zeros((M.shape[0], 9))
    A[:, 0] = M_norm[:, 0] * M_norm[:, 2]
    A[:, 1] = M_norm[:, 0] * M_norm[:, 3]
    A[:, 2] = M_norm[:, 0]
    A[:, 3] = M_norm[:, 1] * M_norm[:, 2]
    A[:, 4] = M_norm[:, 1] * M_norm[:, 3]
    A[:, 5] = M_norm[:, 1]
    A[:, 6] = M_norm[:, 2]
    A[:, 7] = M_norm[:, 3]
    A[:, 8] = 1
    U, S, V = np.linalg.svd(A)
    return np.transpose(T2)@np.reshape(V[np.argmin(S)],(3,-1))@T1

def compute_F_mine(M):
    width = image_size[0] / 2
    height = image_size[1] / 2
    width2 = image_size[2] / 2
    height2 = image_size[3] / 2
    meanval = np.array([width, height, width2, height2])
    M_norm = M - meanval
    M_norm = M_norm / np.array([width, height, width2, height2])

    T1 = np.array([[1 / width, 0, 0], [0, 1 / height, 0], [0, 0, 1]]) @ np.array([[1, 0, -meanval[0]], [0, 1, -meanval[1]], [0, 0, 1]])
    T2 = np.array([[1 / width2, 0, 0], [0, 1 / height2, 0], [0, 0, 1]]) @ np.array([[1, 0, -meanval[2]], [0, 1, -meanval[3]], [0, 0, 1]])

    F_ret = []
    F_error = np.inf
    start_time = time.time()
    while time.time() - start_time <= 4 :
        a = np.arange(M.shape[0])
        np.random.shuffle(a)
        a = a[:8]
        A = np.zeros((8, 9))
        A[:, 0] = M_norm[a, 0] * M_norm[a, 2]
        A[:, 1] = M_norm[a, 0] * M_norm[a, 3]
        A[:, 2] = M_norm[a, 0]
        A[:, 3] = M_norm[a, 1] * M_norm[a, 2]
        A[:, 4] = M_norm[a, 1] * M_norm[a, 3]
        A[:, 5] = M_norm[a, 1]
        A[:, 6] = M_norm[a, 2]
        A[:, 7] = M_norm[a, 3]
        A[:, 8] = 1
        U, S, V = np.linalg.svd(A)
        F_val = np.transpose(T2) @ np.reshape(V[np.argmin(S)], (3, -1)) @ T1
        error = compute_avg_reproj_error.compute_avg_reproj_error(M, F_val)
        if F_error > error:
            F_error = error
            F_ret = F_val

    F_norm = compute_F_norm(M)
    if F_error > compute_avg_reproj_error.compute_avg_reproj_error(M, F_norm):
        return F_norm
    else :
        return F_ret
#1-1
for image_name in image_list:
    print("Average Rprojection Errors ("+image_name[0]+" and "+image_name[1]+")")
    img1 = cv2.imread(image_name[0],cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_name[1],cv2.IMREAD_GRAYSCALE)
    M = np.loadtxt(image_name[2])
    image_size = np.array([img1.shape[1], img1.shape[0], img2.shape[1],img2.shape[0]])
    print("     Raw =",compute_avg_reproj_error.compute_avg_reproj_error(M, compute_F_raw(M)))
    print("     Norm =",compute_avg_reproj_error.compute_avg_reproj_error(M, compute_F_norm(M)))
    print("     Mine =", compute_avg_reproj_error.compute_avg_reproj_error(M, compute_F_mine(M)))


#1-2
for image_name in image_list:
    img1_ori = cv2.imread(image_name[0],cv2.IMREAD_COLOR)
    img2_ori = cv2.imread(image_name[1],cv2.IMREAD_COLOR)
    waitkey_input = ''
    M = np.loadtxt(image_name[2])
    computed_M = compute_F_mine(M)
    while waitkey_input!=ord('q'):
        img1 = np.copy(img1_ori)
        img2 = np.copy(img2_ori)

        a = np.arange(M.shape[0])
        np.random.shuffle(a)

        image_size = np.array([img1.shape[1], img1.shape[0], img2.shape[1], img2.shape[0]])
        #Draw Circle
        cv2.circle(img1, (int(M[a[0],0]), int(M[a[0],1])),5,(0,0,255),2)
        cv2.circle(img1, (int(M[a[1], 0]), int(M[a[1], 1])),5, (0, 255, 0), 2)
        cv2.circle(img1, (int(M[a[2], 0]), int(M[a[2], 1])),5, (255, 0, 0), 2)

        cv2.circle(img2, (int(M[a[0], 2]), int(M[a[0], 3])), 5, (0, 0, 255), 2)
        cv2.circle(img2, (int(M[a[1], 2]), int(M[a[1], 3])), 5, (0, 255, 0), 2)
        cv2.circle(img2, (int(M[a[2], 2]), int(M[a[2], 3])), 5, (255, 0, 0), 2)

        #Right Line

        matrix = np.array([[M[a[0],0], M[a[0],1],1 ], [M[a[1],0], M[a[1],1],1 ], [M[a[2],0], M[a[2],1],1 ]])
        mul = matrix @ np.transpose(computed_M)


        cv2.line(img2, (0, int(-mul[0, 2] / mul[0, 1])),(int(img2.shape[1]), int((-mul[0, 2] - mul[0, 0] * img2.shape[1]) / mul[0, 1])), (0, 0, 255), 1)
        cv2.line(img2, (0, int(-mul[1, 2] / mul[1, 1])),(int(img2.shape[1]), int((-mul[1, 2] - mul[1, 0] * img2.shape[1]) / mul[1, 1])), (0, 255, 0), 1)
        cv2.line(img2, (0, int(-mul[2, 2] / mul[2, 1])),(int(img2.shape[1]), int((-mul[2, 2] - mul[2, 0] * img2.shape[1]) / mul[2, 1])), (255, 0, 0), 1)


        #Left Line


        matrix = np.array([[M[a[0], 2], M[a[0], 3], 1], [M[a[1], 2], M[a[1], 3], 1], [M[a[2], 2], M[a[2], 3], 1]])
        mul = matrix @ computed_M

        cv2.line(img1, (0, int(-mul[0, 2] / mul[0, 1])),(int(img1.shape[1]), int((-mul[0, 2] - mul[0, 0] * img1.shape[1]) / mul[0, 1])), (0, 0, 255), 1)
        cv2.line(img1, (0, int(-mul[1, 2] / mul[1, 1])),(int(img1.shape[1]), int((-mul[1, 2] - mul[1, 0] * img1.shape[1]) / mul[1, 1])), (0, 255, 0), 1)
        cv2.line(img1, (0, int(-mul[2, 2] / mul[2, 1])),(int(img1.shape[1]), int((-mul[2, 2] - mul[2, 0] * img1.shape[1]) / mul[2, 1])), (255, 0, 0), 1)

        img_out=np.zeros((max(img1.shape[0], img2.shape[0]),img1.shape[1] + img2.shape[1],3))
        img_out[0:img1.shape[0], 0:img1.shape[1],:] = img1/255
        img_out[0:img2.shape[0], img1.shape[1]:,:] = img2/255
        cv2.imshow("IMG1",img_out)
        waitkey_input = cv2.waitKey()