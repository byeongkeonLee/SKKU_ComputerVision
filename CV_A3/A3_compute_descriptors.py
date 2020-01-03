import cv2
import numpy as np
# 760512
a=2
N = 1000
D = 1000
quotation = N//(D+1)+1
data = []
center = np.zeros((D,128))
cluster = []
for i in range(N):
    if i==10 or i==100 :
        a -=1
    f = open("./sift/sift100"+"0"*a+str(i),'rb')
    s = f.read()
    tmp = np.reshape(np.frombuffer(s,dtype = np.uint8),(-1,128))
    #if i<D:
    center[i//quotation,:] += np.average(tmp,axis=0)
    cluster.append(np.zeros((tmp.shape[0]),dtype=int))
    data.append(tmp)
    f.close()
#for i in range(N,D):
#    center[i, :] = data[i%N][0]
center = center/quotation

#Kmeans
epoch=0
y = np.zeros((N, D)).astype(np.float32)
while epoch<20:
    for i in range(N):
        if i%10==0:
            print(i)
        for j in range(data[i].shape[0]):
            cluster[i][j] = np.argmin(np.linalg.norm(data[i][j] - center,axis=1))

    #center update
    center_update = np.zeros((D,128))
    center_update_cnt = np.zeros((D,1))
    for i in range(N):
        for j in range(data[i].shape[0]):
            center_update[cluster[i][j],:] += data[i][j]
            center_update_cnt[cluster[i][j]]+=1

    center = center_update / (center_update_cnt+1e-16)
    epoch+=1

    x = np.array([N, D], dtype=np.int32)

    for i in range(N):
        for j in range(data[i].shape[0]):
            y[i, cluster[i][j]] -= np.power(np.linalg.norm(data[i][j]-center[cluster[i][j]]),0.25)/data[i].shape[0]

    #print(y, np.min(y[:,:],axis=0),np.max(y[:,:],axis=0))
    f = open("A3_2014312692.des", 'wb')
    f.write(x.tobytes())
    f.write(y.tobytes())
    f.close()
