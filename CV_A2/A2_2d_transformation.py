import numpy as np
import cv2

(xx, yy) = np.meshgrid(np.arange(801), np.arange(801))
xx -=400
yy -=400
smile = cv2.imread("image/smile.png", cv2.IMREAD_GRAYSCALE)
def get_transformed_image(img, M):
    img = np.ones((801,801))
    (xx, yy) = np.meshgrid(np.arange(smile.shape[1]), np.arange(smile.shape[0]))
    #xx -= smile.shape[1]//2
    #yy -= smile.shape[0] // 2
    M = M @ [[1,0,-smile.shape[1]//2],[0,1,-smile.shape[0]//2],[0,0,1]]
    tmp = np.array(yy * M[1,1] + xx * M[1,0] + M[1,2])
    y_min , y_max = int(np.min(tmp)), int(np.max(tmp))+1
    tmp = np.array(yy * M[0,1] + xx * M[0,0] + M[0,2])
    x_min, x_max = int(np.min(tmp)), int(np.max(tmp))+1

    xx,yy = np.meshgrid(np.arange(x_max-x_min+1), np.arange(y_max-y_min+1))
    yy, xx = yy+y_min, xx+x_min

    Minv = np.linalg.inv(M)
    smile2 = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 2)).astype('int32')
    tmp = np.array(yy * Minv[1, 1] + xx * Minv[1, 0] + Minv[1, 2])
    tmp[tmp>smile.shape[0]-1]=smile.shape[0]-1
    tmp[tmp < 0] = 0
    smile2[:,:,0] = tmp

    tmp = np.array(yy * Minv[0, 1] + xx * Minv[0, 0] + Minv[0, 2])
    tmp[tmp > smile.shape[1] - 1] = smile.shape[1] - 1
    tmp[tmp <0] = 0
    smile2[:,:,1] = tmp

    smile3 = np.zeros((y_max-y_min+1,x_max-x_min+1))
    smile3[:,:] = smile[smile2[:,:,0],smile2[:,:,1]]
    #center=(400+smile.shape[0]//2, 400+smile.shape[1]//2)
    center = (400,400)
    img[(center[0]+yy[:,:])%801, (center[1]+xx[:,:])%801]= smile3[:,:]


    cv2.arrowedLine(img, (0,400),(800,400),(0,0,0),thickness=2,tipLength=0.01)
    cv2.arrowedLine(img, (400,800),(400, 0), (0, 0, 0), thickness=2, tipLength=0.01)
    return img


if __name__ =="__main__":
    img = np.ones((801,801))
    M = np.eye(3)
    while True:
        img = get_transformed_image(img,M)
        cv2.imshow("res", img)
        angle = 5/180*np.pi
        c=cv2.waitKey()
        if c == ord('q'):
            break
        if c== ord('a'):
            M = [[1,0,-5],[0,1,0],[0,0,1]] @ M
        if c== ord('d'):
            M = [[1,0,5],[0,1,0],[0,0,1]] @ M
        if c == ord('w'):
            M = [[1, 0, 0], [0, 1, -5], [0, 0, 1]] @ M
        if c== ord('s'):
            M = [[1,0,0],[0,1,5],[0,0,1]] @ M
        if c== ord('r'):
            M = [[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]] @ M
        if c == ord('R'):
            M = [[np.cos(-angle), np.sin(-angle), 0], [-np.sin(-angle), np.cos(-angle), 0], [0, 0, 1]] @ M
        if c== ord('F'):
            M = [[1,0,0],[0,-1,0],[0,0,1]] @ M
        if c == ord('f'):
            M = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]] @ M
        if c == ord('x'):
            M = [[0.95, 0, 0], [0, 1, 0], [0, 0, 1]] @ M
        if c == ord('X'):
            M = [[1.05, 0, 0], [0, 1, 0], [0, 0, 1]] @ M
        if c == ord('y'):
            M = [[1, 0, 0], [0, 0.95, 0], [0, 0, 1]] @ M
        if c == ord('Y'):
            M = [[1, 0, 0], [0, 1.05, 0], [0, 0, 1]] @ M
        if c == ord('H'):
            M = np.eye(3)