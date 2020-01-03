import numpy as np
import cv2

def my_orb(img):
    orb1 = cv2.ORB_create()
    kp1 = orb1.detect(img, None)
    return orb1.compute(img, kp1)

def my_Matcher(des1, des2):
    matches = []
    row_score1 = np.zeros((len(des1)))
    idx1 = np.zeros((len(des1), 2)).astype('uint64')
    score1 = np.zeros(len(des2))
    for i in range(len(des1)):
        for j in range(len(des2)):
            score1[j] = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
        row_score1[i] = min(score1)
        idx1[i] = (i, np.argmin(score1))

    rank = row_score1.argsort()
    # print("rank (src->dest)",row_score1[rank[:top]])
    # print(idx1[rank[:top]])

    row_score2 = np.zeros((len(des2)))
    idx2 = np.zeros((len(des2), 2)).astype('uint64')
    score2 = np.zeros(len(des1))
    for i in range(len(des2)):
        for j in range(len(des1)):
            score2[j] = cv2.norm(des1[j], des2[i], cv2.NORM_HAMMING)
        row_score2[i] = min(score2)
        idx2[i] = (np.argmin(score2), i)

    rank2 = row_score2.argsort()
    # print("rank2 (dest->src)", row_score2[rank2[:top]])
    # print(idx2[rank2[:top]])

    markedSrc = np.zeros((len(des1) + 1))
    markedDest = np.zeros((len(des2) + 1))
    matches = []
    i = j = 0
    while i < len(des1) and j < len(des2):
        matches.append(cv2.DMatch(idx1[rank[i], 0], idx1[rank[i], 1], row_score1[rank[i]]))
        markedDest[idx1[rank[i], 1]] = markedSrc[idx1[rank[i], 0]] = 1
        i += 1
        j += 1
        if i < len(rank) - 1:
            while (markedSrc[idx1[rank[i], 0]] or markedDest[idx1[rank[i], 1]]):
                i += 1
        if j < len(rank2) - 1:
            while (markedSrc[idx2[rank2[j], 0]] or markedDest[idx2[rank2[j], 1]]) and j < len(rank2) - 1:
                j += 1
    if i < len(des1):
        matches.append(cv2.DMatch(idx1[rank[i], 0], idx1[rank[i], 1], row_score1[rank[i]]))
        #print(i, j, "(", idx1[rank[i], 0], ",", idx1[rank[i], 1], ",", row_score1[rank[i]], ")", idx2[rank2[j], 0],     markedDest[idx2[rank2[j], 1]])
        markedDest[idx1[rank[i], 1]] = markedSrc[idx1[rank[i], 0]] = 1
        i += 1
        if i < len(rank) - 1:
            while (markedSrc[idx1[rank[i], 0]] or markedDest[idx1[rank[i], 1]]):
                i += 1
    if j < len(des2):
        matches.append(cv2.DMatch(idx2[rank2[j], 0], idx2[rank2[j], 1], row_score2[rank2[j]]))
        markedDest[idx2[rank2[j], 1]] = markedSrc[idx2[rank2[j], 0]] = 1
        #print(j)
        j += 1
        if j < len(rank2) - 1:
            while (markedSrc[idx2[rank2[j], 0]] or markedDest[idx2[rank2[j], 1]]):
                j += 1

    return matches,idx1,rank

def my_kp(kp1,kp2,top,idx1,rank):

    kp1pt = np.zeros((top,2))
    kp2pt = np.zeros((top,2))
    for i in range(top):
        kp1pt[i]=kp1[idx1[rank[i],0]].pt
        kp2pt[i]=kp2[idx1[rank[i],1]].pt
    return kp1pt,kp2pt