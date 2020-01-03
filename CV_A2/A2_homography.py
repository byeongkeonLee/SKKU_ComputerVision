import numpy as np
import cv2
import my_function
import time
def compute_homography(srcP, destP):
    meanV = -np.median(srcP,axis=0)
    Xs = srcP + meanV
    scaleV = 1/np.max(np.sqrt(np.power(Xs[:, 0], 2) + np.power(Xs[:, 1], 2))) * np.sqrt(2)
    Ts = np.array([[scaleV,0,0],[0,scaleV,0],[0,0,1]]) @ np.array([[1,0,meanV[0]],[0,1,meanV[1]],[0,0,1]]) @ np.eye(3)
    Xs *= scaleV

    meanV = -np.median(destP, axis=0)
    Xd = destP + meanV
    scaleV = 1 / np.max(np.sqrt(np.power(Xd[:, 0], 2) + np.power(Xd[:, 1], 2))) * np.sqrt(2)
    Td = np.array([[scaleV, 0, 0], [0, scaleV, 0], [0, 0, 1]]) @ np.array([[1, 0, meanV[0]], [0, 1, meanV[1]], [0, 0, 1]]) @ np.eye(3)
    Xd *= scaleV

    srcx = Xs[:,0].reshape(-1,1)
    srcy = Xs[:, 1].reshape(-1,1)
    destx = Xd[:,0].reshape(-1,1)
    desty = Xd[:,1].reshape(-1,1)
    A = np.zeros((srcP.shape[0],2,9))
    A[:,0,0] = -srcx[:,0]
    A[:, 0, 1] = -srcy[:, 0]
    A[:, 0, 2] = -1
    A[:, 0, 6] = (srcx * destx)[:,0]
    A[:, 0, 7] = (srcy * destx)[:,0]
    A[:,0,8] = destx[:,0]
    A[:,1,3] = -srcx[:,0]
    A[:, 1, 4] = -srcy[:, 0]
    A[:, 1, 5] = -1
    A[:, 1, 6] = (srcx * desty)[:,0]
    A[:, 1, 7] = (srcy * desty)[:,0]
    A[:,1,8] = desty[:,0]
    A = A.reshape(1,-1,9).squeeze(axis=0)
    u,sigma,v=np.linalg.svd(A)
    H = v[np.argmin(sigma)].reshape(3,3)
    H = np.linalg.inv(Td)@H@Ts
    H = H/H[2,2]
    return H

def compute_homography_ransac(srcP, destP,th):
    #print(srcP,destP)
    i=0
    maxcnt=-1
    ret=None
    start= time.time()
    while time.time()-start <=3:
        i+=1
        a = np.arange(len(srcP))
        np.random.shuffle(a)

        a = a[:4]
        #a = [0,1,2,3,4]
        H = compute_homography(srcP[a],destP[a])
        tmp = (np.c_[srcP, np.ones((len(srcP)))]).T

        calculated = H@tmp
        #calculated = np.einsum('ik,kj->ij',H,tmp)
        calculated[calculated[:,:]==0]=1e-17

        calculated = (calculated / calculated[2]).T
        #print(calculated[:,0:2],destP[:,0:2])
        distance = np.sqrt(np.power(calculated[:,0]-destP[:,0],2)+np.power(calculated[:,1]-destP[:,1],2))

        cnt = np.sum(distance<th)
        #print(distance, cnt,"A")

        if maxcnt < cnt:
            maxcnt = cnt
            ret = H
    return ret

if __name__ =="__main__":

    #####################################################parameter###############################################
    top = 10  # the number of matchers for 2-1
    kpnum = 17 # homography the number of kp 2-2
    kpnum2 = 30 # ransac the number of kp 2-4
    th = 14 # threshold of ransac 2-4
    kpnum3 = 100 # diamond image ransac the number of kp 2-5
    th2 = 20 # threshold of ransac 2-5
    blendingwidth = 80 # width of diamond blending area 2-5
    """
    parameter 
    """
    img1= cv2.imread("image/cv_desk.png",cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("image/cv_cover.jpg", cv2.IMREAD_GRAYSCALE)

    kp1,des1 = my_function.my_orb(img1)
    kp2,des2 = my_function.my_orb(img2)

    matches,idx1,rank = my_function.my_Matcher(des1,des2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:top], None, flags=2)
    cv2.imshow("2_1-top10matches",img3)

    """
    2-2
    """

    length = len(matches)
    srcP = np.zeros((length,2))
    destP = np.zeros((length, 2))
    for i in range(length):
        destP[i] = np.array([kp1[matches[i].queryIdx].pt])
        srcP[i] = np.array([kp2[matches[i].trainIdx].pt])

    """
    2-3
    """
    H=compute_homography(srcP[:kpnum],destP[:kpnum])
    img_result = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]))
    cv2.imshow("homography_with_normalization",img_result)

    img_result += np.where(img_result==0,img1,0)
    cv2.imshow("homography_with_normalization_COVER",img_result)

    """
    2-4
    """
    H = compute_homography_ransac(srcP[:kpnum2],destP[:kpnum2],th)
    img_result = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    cv2.imshow("homography_with_ransac",img_result)
    img_result += np.where(img_result == 0, img1, 0)
    cv2.imshow("homography_with_ransac_COVER",img_result)

    harry = cv2.imread("./image/hp_cover.jpg",cv2.IMREAD_GRAYSCALE)
    harry_re = []
    harry_re=cv2.resize(harry,dsize=(img2.shape[1],img2.shape[0]),interpolation=cv2.INTER_AREA)
    harry_result = cv2.warpPerspective(harry_re,H,(img1.shape[1],img1.shape[0]))
    cv2.imshow("harrypoter_ransac",harry_result)
    harry_result += np.where(harry_result==0,img1,0)
    cv2.imshow("harrypoter_ransac_COVER", harry_result)

    #print(H)

    """
    2-5
    """
    dia1 = cv2.imread("image/diamondhead-10.png", cv2.IMREAD_GRAYSCALE)
    dia2 = cv2.imread("image/diamondhead-11.png", cv2.IMREAD_GRAYSCALE)
    kp1, des1 = my_function.my_orb(dia1)
    kp2, des2 = my_function.my_orb(dia2)
    matches, idx1, rank = my_function.my_Matcher(des1, des2)

    length = len(matches)
    srcP = np.zeros((length, 2))
    destP = np.zeros((length, 2))
    for i in range(length):
        destP[i] = np.array([kp1[matches[i].queryIdx].pt])
        srcP[i] = np.array([kp2[matches[i].trainIdx].pt])

    H = compute_homography_ransac(srcP[:kpnum3], destP[:kpnum3], th2)
    img_result = cv2.warpPerspective(dia2, H, (dia1.shape[1]+1000, dia1.shape[0]))

    img1 = np.ones((dia1.shape[0],dia1.shape[1]+1000))
    y_min = np.min(np.where(img_result!=0)[0])
    x_max = np.max(np.where(img_result!=0)[1])
    y_max =np.max(np.where(img_result!=0)[0])


    boundary = 0
    img1[:,:dia1.shape[1]-boundary] = dia1[:,:dia1.shape[1]-boundary]/255
    img1[:, dia1.shape[1]:] = img_result[:, dia1.shape[1]:] / 255
    for i in range(-boundary,0):
        img1[:, dia1.shape[1] + i] = dia1[:, dia1.shape[1] + i]/255 * (-i/boundary) + img_result[:, dia1.shape[1]+i]/255 * ((i+boundary)/boundary)

    out1=img1[y_min:y_max, :x_max]
    x_max2 = np.min(np.where(out1 == 0)[1])
    out1=img1[:,:x_max2]
    cv2.imshow("Diamond_stiching",out1)

    boundary = blendingwidth
    img2 = np.ones((dia1.shape[0], dia1.shape[1] + 1000))
    img2[:, :dia1.shape[1] - boundary] = dia1[:, :dia1.shape[1] - boundary] / 255
    img2[:, dia1.shape[1]:] = img_result[:, dia1.shape[1]:] / 255
    for i in range(-boundary, 0):
        img2[:, dia1.shape[1] + i] = dia1[:, dia1.shape[1] + i] / 255 * (-i / boundary) + img_result[:,dia1.shape[1] + i] / 255 * ((i + boundary) / boundary)

    out2 = img2[y_min:y_max, :x_max]
    x_max2 = np.min(np.where(out2 == 0)[1])
    cv2.imshow("Diamond_blending", out2[:, :(x_max2+out2.shape[1])//2])
    #cv2.imshow("Diamond_Blending", out2)
    cv2.waitKey()



