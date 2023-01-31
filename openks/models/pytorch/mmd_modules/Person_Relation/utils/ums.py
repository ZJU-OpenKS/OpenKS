
import torch
import torch.nn as nn
import numpy as np
import os
import math
NUM_ENLARGE_SIZE = 100

def euclidean(p1, p2):
    distX = abs(p1["x"] - p2["x"])
    distXSquare = distX * distX

    distY = abs(p1["y"] - p2["y"])
    distYSquare = distY * distY

    distT = abs(p1["time"] - p2["time"])
    distTSquare = distT * distT * 100

    distSquare = (distXSquare+distYSquare) * (1+distTSquare)
    return math.sqrt(distSquare)

def triangular(p1, p2):
    distX = abs(p1["x"] - p2["x"])
    distXSquare = distX*distX

    distY = abs(p1["y"]-p2["y"])
    distYSquare = distY * distY

    distT = abs(p1["time"] - p2["time"])
    distTSquare = distT * distT * 100

    distSquare = (distXSquare + distYSquare) * (1 + distTSquare)
    return math.sqrt(2*(distSquare))

def angle(p1, p2):
    return math.atan2(p2["x"] - p1["x"], p2["y"] - p1["y"])

def near_equal(P1, P2):
    """
    简单计算点之间的相似度，适用于点数少的情况
    :param P1:
    :param P2:
    :return:
    """
    if P1["time"] - 5 > P2["time"] or P1["time"] + 5 < P2["time"]:
        return 0
    num = 0.0001 * NUM_ENLARGE_SIZE
    if P1["x"] - num > P2["x"] or P1["x"] + num < P2["x"]:
        return 0
    if P1["y"] - num > P2["y"] or P1["y"] + num < P2["y"]:
        return 0
    return 1

def createEllipticalTrajectory(t):
    """
    获得的边
    :param t: list
    :return:
    """
    i = 0
    T = []
    for i in range(len(t)-1):
        p1 = t[i]
        p2 = t[i+1].copy()

        num = 0.0001 * 20 * NUM_ENLARGE_SIZE
        if abs(p1["x"] - p2["x"]) > num:
            if p1["x"] > p2["x"]:
                p2["x"] = p1["x"] - num
            else:
                p2["x"] = p1["x"] + num
        if abs(p1["y"] - p2["y"]) > num:
            if p1["y"] > p2["y"]:
                p2["y"] = p1["y"] - num
            else:
                p2["y"] = p1["y"] + num
        if abs(p1["time"] - p2["time"]) > 30:
            if p1["time"] > p2["time"]:
                p2["time"] = p1["time"] - 30
            else:
                p2["time"] = p1["time"] + 30

        x = (p1["x"] + p2["x"]) / 2
        y = (p1["y"] + p2["y"]) / 2
        time = (p1["time"] + p2["time"]) / 2

        fociDistance = euclidean(p1, p2)
        majorAxis = triangular(p1, p2) + 1

        fociDistanceSquare = fociDistance * fociDistance
        majorAxisSquare = majorAxis * majorAxis

        minorAxis = math.sqrt(majorAxisSquare - fociDistanceSquare)

        angleO = angle(p1, p2)
        e = {}
        e["Eid"] = i
        e["StartTime"] = p1["time"]
        e["EndTime"] = p2["time"]
        e["time"] = time
        e["CenterX"] = x
        e["CenterY"] = y
        e["x"] = x
        e["y"] = y
        e["F1"] = p1
        e["F2"] = p2
        e["SemiMajorAxis"] = majorAxis / 2
        e["MajorAxis"] = majorAxis
        e["MinorAxis"] = minorAxis
        e["Angle"] = angleO
        e["Eccentricity"] = fociDistance
        T.append(e)
    return T


def ums_dist(T1, T2):
    """
    计算ums相似度
    T1,T2先设定为List类型
    :param T1:
    :param T2:
    :return:
    """
    aSet, bSet = [], []
    n = len(T1)
    m = len(T2)
    if n < 3 or m < 3:
    #XXX Exceptions for when the trajectories have less than 3 points, but the points are equal
        if n == 1 and m == 1:
            return near_equal(T1[0], T2[0])
        elif n == 2 and m == 2:
            return max((near_equal(T1[0], T2[0]) + near_equal(T1[1], T2[1])) / 2,
                       (near_equal(T1[1], T2[0]) + near_equal(T1[0], T2[1])) / 2)
        elif n == 1 and m == 2:
            return (near_equal(T1[0], T2[0]) + near_equal(T1[0], T2[1])) / 2
        elif n == 2 and m == 1:
            return (near_equal(T1[0], T2[0]) + near_equal(T1[1], T2[0])) / 2
        return 0

    # Create Elliptical Trajectories(it can be performed offline speeding
    # up multiple similarity computations)
    E1 = createEllipticalTrajectory(T1)
    E2 = createEllipticalTrajectory(T2)

    # Compute Alikeness and Shareness
    aMatchSet = {}
    bMatchSet = {}

    shr1 = np.zeros(n, dtype=np.float16)
    shr2 = np.zeros(m, dtype=np.float16)

    for i in range(n-1):
        e1 = E1[i]
        for j in range(m - 1):
            e2 = E2[j]
            if (euclidean(e1, e2) <= (e1["MajorAxis"] / 2) + (e2["MajorAxis"] / 2)):
                p1shr = getShareness(e1["F1"], e2)
                if p1shr > 0:
                    aSet.append(j)
                    if i not in aMatchSet:
                        aMatchSet[i] = []
                        aMatchSet[i].append(j)
                    else:
                        aMatchSet[i].append(j)
                    if abs(shr1[i] - 1.0) > 0.000001:
                        shr1[i] = p1shr if p1shr > shr1[i] else shr1[i]

                p2shr = getShareness(e1["F2"], e2)
                if p2shr > 0:
                    aSet.append(j)
                    if i+1 not in aMatchSet:
                        aMatchSet[i+1] = []
                        aMatchSet[i+1].append(j)
                    else:
                        aMatchSet[i+1].append(j)
                    if abs(shr1[i+1] - 1.0) > 0.000001:
                        shr1[i+1] = p2shr if p2shr > shr1[i+1] else shr1[i+1]

                q1shr = getShareness(e2["F1"], e1)
                if q1shr > 0:
                    bSet.append(i)
                    if j not in bMatchSet:
                        bMatchSet[j] = []
                        bMatchSet[j].append(i)
                    else:
                        bMatchSet[j].append(i)
                    if abs(shr2[j] - 1.0) > 0.000001:
                        shr2[j] = q1shr if q1shr > shr2[j] else shr2[j]

                q2shr = getShareness(e2["F2"], e1)
                if q2shr > 0:
                    bSet.append(i)
                    if j+1 not in bMatchSet:
                        bMatchSet[j+1] = []
                        bMatchSet[j+1].append(i)
                    else:
                        bMatchSet[j+1].append(i)
                    if abs(shr2[j+1] - 1.0) > 0.000001:
                        shr2[j+1] = q2shr if q2shr > shr2[j+1] else shr2[j+1]

    # Compute Continuity
    aContinuity = np.zeros(n, dtype=np.int32)
    bContinuity = np.zeros(m, dtype=np.int32)
    aResult = 0.
    bResult = 0.

    for j in range(n):
        #matchingSet = aMatchSet[j]
        if j == 0:
            aContinuity[j] = -1 if j not in aMatchSet else min(aMatchSet[j])
        else:
            aContinuity[j] = -1 if j not in aMatchSet else getContinuityValue(aContinuity[j - 1], aMatchSet[j])

        if aContinuity[j] != -1:
            if j == 0:
                aResult += 1
            elif aContinuity[j] >= aContinuity[j - 1]:
                aResult += 1

    for j in range(m):
        #matchingSet = bMatchSet[j]
        if j == 0:
            bContinuity[j] = -1 if j not in bMatchSet else min(bMatchSet[j])
        else:
            bContinuity[j] = -1 if j not in bMatchSet else getContinuityValue(bContinuity[j - 1], bMatchSet[j])

        if bContinuity[j] != -1:
            if j == 0:
                bResult += 1
            elif bContinuity[j] >= bContinuity[j - 1]:
               bResult += 1

    continuity = (aResult / n) * (bResult / m)
    sum1 = 0
    sum3 = 0.

    for j in range(n):
        if shr1[j] > 0.0:
            sum1 += 1
            sum3 += shr1[j]

    sum2 = 0
    sum4 = 0.

    for j in range(m):
        if shr2[j] > 0.0:
            sum2 += 1
            sum4 += shr2[j]

    alikeness1 = float(sum1) / n
    alikeness2 = float(sum2) / m

    shareness1 = sum3 / n
    shareness2 = sum4 / m

    alikeness = (alikeness1 * alikeness2)
    shareness = 0.5 * (shareness1 + shareness2)

    similarity = (0.5 * (alikeness + shareness)) * continuity
    #print("alikeness", alikeness)
    #print("shareness", shareness)
    #print("continuity", continuity)
    #print("similarity", similarity)
    return similarity

def getContinuityValue(lastValue, matchingList):
    matchingList1 = matchingList.copy()
    matchingList1.sort()
    for i in matchingList1:
        if i >= lastValue:
            return i
    return -1

def getShareness(p1, e):
    """
    p1是一个点，e是一条边
    :param p1:
    :param e:
    :return:
    """

    centerX = e["CenterX"]
    centerY = e["CenterY"]
    angle = e["Angle"]
    cos = math.cos(angle)
    sin = math.sin(angle)

    semiMinorAxis = e["MinorAxis"] / 2
    semiMajorAxis = e["MajorAxis"] / 2

    semiMinorAxisSquare = semiMinorAxis * semiMinorAxis
    semiMajorAxisSquare = semiMajorAxis * semiMajorAxis

    cx = centerX
    cy = centerY
    px = p1["x"]
    py = p1["y"]

    aCos = (cos * (px - cx) - sin * (py - cy))
    a = aCos * aCos
    bSin = (sin * (px - cx) + cos * (py - cy))
    b = bSin * bSin

    eIn = (a / semiMinorAxisSquare) + (b / semiMajorAxisSquare)

    if eIn > 1:
        return 0.

    distF1 = euclidean(p1, e["F1"])
    distF2 = euclidean(p1, e["F2"])
    distC = euclidean(p1, e)

    hypotenuse = (semiMinorAxisSquare + semiMajorAxisSquare) / e["MajorAxis"]
    min_num = 1.
    min_num = min(min(distF1, distF2), distC) / hypotenuse
    #if abs(min) < 0.000001:
    #    return 1

    return 1 - min_num


