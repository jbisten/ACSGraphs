import re
import numpy as np
import scipy
from plyfile import PlyData
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import torch
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import cuml
from cuml.svm import SVC
import cupy as cp


class plyStruct():
    "Class that process poly data and create isomap"

    def loadPolyData(self, subpath):
        "Loading ply file and its stream index"
        ply = PlyData.read(subpath)
        elemList = []
        elemDict = {}
        self.vprt = []
        self.plyData = []
        self.idx = []
        self.feDict = {}
        for i in range(len(ply.elements)):
            elemList.append(ply.elements[i].name)
            tmplist = []
            for j in range(len(ply.elements[i].properties)):
                tmplist.append(ply.elements[i].properties[j].name)
            elemDict[ply.elements[i].name] = tmplist

        if 'vertices' in elemDict.keys():
            prts = elemDict['vertices']

            # print(prts)
            for prt in prts:
                # print(prt)
                self.plyData.append(ply['vertices'].data[prt])
                self.vprt.append(prt)
                # nprt="self."+prt
                self.feDict[prt] = np.asarray(ply['vertices'].data[prt])
                # exec("{0} = {1}".format(nprt,tmpFe))
            self.plyData = np.asarray(self.plyData)
            self.plyData = self.plyData.T
        if 'fiber' in elemDict.keys():
            self.idx = ply['fiber'].data['endindex']
            self.idxLen = np.copy(self.idx)
            self.idxLen[1:] = self.idx[1:] - self.idxLen[:-1]

def kdInterpolate(p1, p2, v1, th=4):
    kdt = KDTree(p1, leaf_size=40, metric='euclidean')
    vertextMask = list()
    v2 = list()
    for vertex in tqdm(p2, total=len(p2), desc='Interpolating vertices'):
        nvertex = np.array([vertex[0], vertex[1], vertex[2]])
        nV = 0
        dist, ind = kdt.query(nvertex.reshape(1, -1), k=4)
        VtoCDt = []
        for distj in range(0, 4):
            VtoCDt.append(dist[0, distj])
        VtoCDt = np.asarray(VtoCDt)
        if dist.min() != 0:
            maxdist = dist.max()
            dist = maxdist / dist[:]
            nV = 0
            for j in range(0, 4):
                nV += v1[ind[0, j]] * dist[0, j]
            nV = nV / np.sum(dist)
        else:
            nV = v1[ind[0, dist.argmin()]]
        v2.append(nV)
        if VtoCDt.min() < th:
            vertextMask.append(1)
        else:
            vertextMask.append(-1)

    return v2, vertextMask


def fastkdInterpolate(p1, p2, v1, th=4):
    kdt = KDTree(p1, leaf_size=40, metric='euclidean')
    dist, ind = kdt.query(p2, k=4)

    # Vectorized distance inversion and weight calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        maxdist = dist.max(axis=1, keepdims=True)
        weights = maxdist / dist
        weights[dist == 0] = 0  # Explicitly handle zero distances
    
    # Interpolated values calculation
    v1_neighbors = v1[ind]  # Shape: (num_points, 4, v1_dimension)

    # Check if v1 is 1D or 2D
    if v1.ndim == 1:
        v1_neighbors = v1_neighbors[..., np.newaxis]  # Shape: (num_points, 4, 1)

    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights /= weights_sum

    weights = weights[:, :, np.newaxis]  # Shape: (num_points, 4, 1)

    # Interpolated values
    interpolated_values = np.sum(v1_neighbors * weights, axis=1)

    # Vertex mask
    vertex_mask = np.where(np.min(dist, axis=1) < th, 1, -1)

    # Scale interpolated values to the range [0, 1]
    interpolated_values_min = interpolated_values.min()
    interpolated_values_max = interpolated_values.max()
    interpolated_values = (interpolated_values - interpolated_values_min) / (interpolated_values_max - interpolated_values_min)

    return np.array(interpolated_values), np.array(vertex_mask)


def writePly(alldata, allidx, filename):
    header = "comment DTI Tractography, produced by fiber-track\n"
    header += "element vertices {0}\n".format(alldata.shape[0])
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "element fiber {0}\n".format(len(allidx))
    header += "property int endindex\n"
    header += "end_header\n"
    with open(filename, 'w') as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(header)
        for plyvertex in alldata:
            file.write("{0} {1} {2}\n".format(plyvertex[0], plyvertex[1], plyvertex[2]))
        for streamId in allidx:
            file.write("{0}\n".format(streamId))


def writePlyFeatures(alldata, allidx, feNames, feList, filename):
    assert len(feNames) == len(feList)
    header = "comment DTI Tractography, produced by fiber-track\n"
    header += "element vertices {0}\n".format(alldata.shape[0])
    header += "property float x\nproperty float y\nproperty float z\n"
    for fe in feNames:
        header += "property float " + fe + "\n"
    header += "element fiber {0}\n".format(len(allidx))
    header += "property int endindex\n"
    header += "end_header\n"
    with open(filename, 'w') as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(header)
        for vid, plyvertex in enumerate(alldata):
            file.write("{0} {1} {2} ".format(plyvertex[0], plyvertex[1], plyvertex[2]))
            for fe in feList:
                file.write("{0} ".format(fe[vid]))
            file.write("\n")
        for streamId in allidx:
            file.write("{0}\n".format(streamId))


def writeMaskply(alldata, allidx, mask, filename):
    nidx = np.zeros(allidx.shape, dtype=int)
    nidx[1:] = allidx[:-1]
    didx = allidx - nidx
    numIdx = len(mask[mask == 1])

    header = "comment DTI Tractography, produced by fiber-track\n"
    header += "element vertices {0}\n".format(didx[mask == 1].sum())
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "element fiber {0}\n".format(numIdx)
    header += "property int endindex\n"
    header += "end_header\n"
    with open(filename, 'w') as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(header)

        lIdx = 0
        maskid = 0
        for cIdx in allidx:
            if mask[maskid] == 1:
                for plyvertex in alldata[lIdx:cIdx, :]:
                    file.write("{0} {1} {2}\n".format(plyvertex[0], plyvertex[1], plyvertex[2]))
            maskid += 1
            lIdx = np.copy(cIdx)
        #####        
        didx = didx[mask == 1]
        didx = didx.cumsum()
        for streamId in didx:
            file.write("{0}\n".format(streamId))


def writeMaskply2(alldata, allidx, mask, filename):
    nidx = np.zeros(allidx.shape, dtype=int)
    nidx[1:] = allidx[:-1]
    didx = allidx - nidx
    numIdx = len(mask[mask == 1])

    header = "comment DTI Tractography, produced by fiber-track\n"
    header += "element vertices {0}\n".format(didx[mask == 1].sum())
    header += "property float x\nproperty float y\nproperty float z\nproperty float wx\nproperty float wy\nproperty float wz\nproperty float m\nproperty float f\nproperty float a\nproperty float r\n"
    header += "element fiber {0}\n".format(numIdx)
    header += "property int endindex\n"
    header += "end_header\n"
    with open(filename, 'w') as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(header)
        lIdx = 0
        maskid = 0
        for cIdx in allidx:
            if mask[maskid] == 1:
                for plyvertex in alldata[lIdx:cIdx, :]:
                    file.write(
                        "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(plyvertex[0], plyvertex[1], plyvertex[2],
                                                                           plyvertex[3], plyvertex[4], plyvertex[5],
                                                                           plyvertex[6], plyvertex[7], plyvertex[8],
                                                                           plyvertex[9]))
            maskid += 1
            lIdx = np.copy(cIdx)
        #####        
        didx = didx[mask == 1]
        didx = didx.cumsum()
        for streamId in didx:
            file.write("{0}\n".format(streamId))


def getMasked(plyData, idx, mask):
    nPly = np.empty((0, plyData.shape[1]))
    nIdx = list()
    lIdx = 0

    for maskid, cIdx in enumerate(idx):
        if mask[maskid] == 1:
            nPly = np.vstack((nPly, plyData[lIdx:cIdx, :]))
            nIdx.append(cIdx - lIdx)
        lIdx = np.copy(cIdx)
    nIdx = np.cumsum(nIdx)
    return nPly, nIdx


def maskPly(plyData, idx, num):
    idx = np.asarray(idx)
    if num < idx.shape[0]:
        mask = np.zeros(shape=idx.shape[0], dtype=int)
        mask[np.random.choice(idx.shape[0], num, replace=False)] = 1
        fiberLength = np.zeros(shape=idx.shape[0], dtype=int)
        fiberLength[1:] = idx[:-1]
        fiberLength = idx - fiberLength
        vExcludeNum = np.sum(fiberLength[mask == 0])
        numVertex = np.sum(fiberLength[mask == 1])
        fiberLength = fiberLength[mask == 1]
        fiberLength = np.cumsum(fiberLength)
        numIdx = len(mask[mask == 1])
        lv = 0
        nPly = np.empty((0, 3))
        for maskid, idx in enumerate(idx):
            if mask[maskid] == 1:
                nPly = np.vstack((nPly, plyData[lv:idx]))
            lv = idx
        return nPly, fiberLength
    else:
        return plyData, idx


def subSamplePly(plyData, idx, num):
    idx = np.asarray(idx)
    if num < idx.shape[0]:
        mask = np.zeros(shape=idx.shape[0], dtype=int)
        maskV = np.zeros(shape=plyData.shape[0], dtype=int)
        mask[np.random.choice(idx.shape[0], num, replace=False)] = 1
        fiberLength = np.zeros(shape=idx.shape[0], dtype=int)
        fiberLength[1:] = idx[:-1]
        fiberLength = idx - fiberLength
        vExcludeNum = np.sum(fiberLength[mask == 0])
        numVertex = np.sum(fiberLength[mask == 1])
        fiberLength = fiberLength[mask == 1]
        fiberLength = np.cumsum(fiberLength)
        assert num == len(mask[mask == 1])
        lv = 0
        nPly = np.empty((0, 3))
        for maskid, cidx in enumerate(idx):
            if mask[maskid] == 1:
                nPly = np.vstack((nPly, plyData[lv:cidx]))
                maskV[lv:cidx] = 1
            lv = cidx
        return nPly, fiberLength, maskV
    else:
        maskV = np.ones(shape=plyData.shape[0], dtype=int)
        return plyData, idx, maskV


def filterPly(ply, cord, lBound, uBound):
    vArr = np.copy(ply.plyData)
    mask = np.ones(shape=vArr.shape[0], dtype=int)
    mask[vArr[:, cord] > uBound] = -1
    mask[vArr[:, cord] < lBound] = -1

    exV = np.zeros((len(ply.idx)), dtype=np.int)
    nidx = np.copy(ply.idx)

    nidx[1:] = ply.idx[1:] - nidx[:-1]

    lIdx = 0
    for i, cIdx in enumerate(ply.idx):
        # exV[i]=len(mask[mask[lIdx:cIdx]==-1])
        exV[i] = len(np.where(mask[lIdx:cIdx] == -1)[0])
        lIdx = np.copy(cIdx)

    nidx = nidx - exV
    nidx = np.asarray(nidx)
    nidx = nidx[nidx != 0]
    nidx = np.cumsum(nidx)
    # print("len(exV),len(nidx)",len(exV),len(nidx))
    print("Number of excluded  point:", np.sum(exV))
    vArr = vArr[mask == 1, :]
    # print("nidx[-1],vArr.shape[0]",nidx[-1],vArr.shape[0])
    assert nidx[-1] == vArr.shape[0]
    return vArr, nidx, mask


def extractBin(plyData, numBin, idx, feList, mask=True):
    FirstBin = True
    binSpace = np.linspace(0.0, 1.0, numBin + 1)
    binSpace[0] = min(0, plyData[:, 10].min()) - 0.1
    binSpace[-1] = max(1, plyData[:, 10].max()) + 0.1
    for i in range(0, binSpace.size - 1):
        tmparray = plyData[(plyData[:, idx] < binSpace[i + 1]) & (plyData[:, idx] >= binSpace[i]), :]
        if mask == True:
            tmparray = tmparray[tmparray[:, -1] == 1, :]
        binArray = list()
        if tmparray.size > 0:
            for fidx in feList:
                if fidx < plyData.shape[1]:
                    binArray.append(tmparray[:, fidx].mean())
            binArray.append(tmparray.shape[0])
        else:
            for fidx in feList:
                if fidx < plyData.shape[1]:
                    binArray.append(-1)
            binArray.append(0)  # point count
        binArray = np.asarray(binArray)
        if (FirstBin == False):
            feArray = np.vstack((feArray, binArray))
        else:
            feArray = np.copy(binArray)
            FirstBin = False
    return feArray


def NormalizeArr(data, m=[], v=[]):
    x = np.copy(data)
    if len(m) != x.shape[1] or len(v) != x.shape[1]:
        m = np.mean(x, axis=0)
        v = np.std(x, axis=0) + 0.0000001
    x = x - m
    x = x / v
    return x


def feNameToIdx(feNames, features):
    feIdx = list()
    nFeNames = list()
    for fe in features:
        if fe in feNames:
            feIdx.append(feNames.index(fe))
            nFeNames.append(fe)
    return feIdx, nFeNames


def storeData(lList, rList, lDict, rDict, numBin, features, cordName='iso'):
    assert len(lList) == len(rList)
    # Dict[Name]=[Ply,idx,feNames]
    feArraysL = list()
    feArraysR = list()
    feLDict = {}
    feRDict = {}
    for subID, lName in enumerate(lList):
        rName = rList[subID]
        # feList=[6,7,8,9]
        match1 = re.search(r'/\d+/', lName)
        lSub = match1.group().replace('/', '')
        match2 = re.search(r'/\d+/', rName)
        rSub = match2.group().replace('/', '')
        assert lSub == rSub
        # print ("lSub,rSub",lSub,rSub)
        if lName in lDict.keys():
            feList, lDictFe = feNameToIdx(lDict[lName][2], features)
            if cordName in lDict[lName][2]:
                cord = lDict[lName][2].index(cordName)
            else:
                cord = -1
            assert cord >= 0
            lBins = extractBin(lDict[lName][0], numBin, cord, feList)
        else:
            lPly = plyStruct()
            lPly.loadPolyData(lName.rstrip())
            feList, lDictFe = feNameToIdx(lPly.vprt, features)
            if cordName in lPly.vprt:
                cord = lPly.vprt.index(cordName)
            else:
                cord = -1
            assert cord >= 0
            lBins = extractBin(lPly.plyData, numBin, cord, feList)

        if rName in rDict.keys():
            feList, rDictFe = feNameToIdx(rDict[rName][2], features)
            if cordName in rDict[rName][2]:
                cord = rDict[rName][2].index(cordName)
            else:
                cord = -1
            assert cord >= 0
            rBins = extractBin(rDict[rName][0], numBin, cord, feList)
        else:
            rPly = plyStruct()
            rPly.loadPolyData(rName.rstrip())
            feList, rDictFe = feNameToIdx(rPly.vprt, features)
            if cordName in rPly.vprt:
                cord = rPly.vprt.index(cordName)
            else:
                cord = -1
            assert cord >= 0
            rBins = extractBin(rPly.plyData, numBin, cord, feList)

        # lBins,rBins shape -> numBin * feNum

        for fidx in range(0, len(lDictFe)):
            assert rDictFe[fidx] == lDictFe[fidx]

        lDictFe.append('cs')
        rDictFe.append('cs')
        assert len(lDictFe) == len(rDictFe)
        feArraysL.append(lBins)
        feLDict[lSub] = [lBins, lDictFe]
        feArraysR.append(rBins)
        feRDict[rSub] = [rBins, rDictFe]
    # np.save(open(lBundlePath, "wb"),  csArraysL)
    # np.save(open(rBundlePath, "wb"),  csArraysR)
    return np.asarray(feArraysL), np.asarray(feArraysL), feLDict, feRDict


def storeDataBundeMAP(sList, sDict, numBin, features, cordName='iso'):
    # Dict[Name]=[Ply,idx,feNames]
    feArrays = list()
    feDict = {}
    for subID, sName in enumerate(sList):
        # sName=sList[subID]
        # feList=[6,7,8,9]
        match1 = re.search(r'/\d+/', sName)
        sSub = match1.group().replace('/', '')
        # print ("lSub,rSub",lSub,rSub)
        if sName in sDict.keys():
            feList, sDictFe = feNameToIdx(sDict[sName][2], features)
            if cordName in sDict[sName][2]:
                cord = sDict[sName][2].index(cordName)
            else:
                cord = -1
            assert cord >= 0
            sBins = extractBin(sDict[sName][0], numBin, cord, feList)
        else:
            print(sName, "doesn't exist in dict and loading from disk")
            sPly = plyStruct()
            sPly.loadPolyData(sName.rstrip())
            feList, sDictFe = feNameToIdx(sPly.vprt, features)
            if cordName in sPly.vprt:
                cord = sPly.vprt.index(cordName)
            else:
                cord = -1
            assert cord >= 0
            sBins = extractBin(sPly.plyData, numBin, cord, feList)

        # lBins,rBins shape -> numBin * feNum

        sDictFe.append('cs')
        feArrays.append(sBins)
        feDict[sSub] = [sBins, sDictFe]
    return np.asarray(feArrays), feDict


def fileToList(inputfile):
    fileList = list()
    with open(inputfile) as f:
        for subpath in f:
            fileList.append(subpath.rstrip())
    return fileList


def concatenatePly(inputfile):
    alldata = np.empty([0, 3])
    allidx = np.empty([0], dtype=int)
    vNum = list()
    idxNum = list()
    lidxcount = 0
    ldatacount = 0
    with open(inputfile) as f:
        for subpath in f:
            tmpPly = plyStruct()
            print(subpath.rstrip())
            tmpPly.loadPolyData(subpath.rstrip())
            vNum.append(tmpPly.plyData.shape[0])
            idxNum.append(len(tmpPly.idx))
            allidx = np.concatenate((allidx, tmpPly.idx))
            alldata = np.concatenate((alldata, tmpPly.plyData[:, 0:3]))

            allidx[lidxcount:allidx.shape[0]] += ldatacount
            ldatacount = alldata.shape[0]
            lidxcount = allidx.shape[0]
    return alldata, allidx, vNum, idxNum

def covMatrixOptimized(alldata, allidx):
    streamFeatures = []
    indices = [(allidx[i-1], allidx[i]) for i in range(1, allidx.shape[0])]

    for start, end in tqdm(indices, desc='Computing streamline features', total=len(indices)):
        currentSL = alldata[start:end, 0:3]
        means = np.mean(currentSL, axis=0)
        cov = np.cov(currentSL.T)
        
        if np.linalg.det(cov) != 0:
            srcov = scipy.linalg.sqrtm(cov)
            tmpFeatures = np.concatenate((means, srcov[np.triu_indices(3)]))
            streamFeatures.append(tmpFeatures)
        else:
            print("warning: np.linalg.det(cov)=0")

    streamFeatures = np.array(streamFeatures)
    return streamFeatures


def covMatrix(alldata, allidx):
    streamFeatures = np.empty([0, 9])
    lastindex = 0

    # Potentially Parallize on GPU
    for i in tqdm(range(allidx.shape[0]), desc='Computing streamline features', total=allidx.shape[0]):
        currentSL = np.copy(alldata[lastindex:allidx[i] - 1, 0:3])
        m1 = np.mean(currentSL[:, 0])
        m2 = np.mean(currentSL[:, 1])
        m3 = np.mean(currentSL[:, 2])
        cov = np.cov(currentSL.T)
        if np.linalg.det(cov) != 0:
            srcov = scipy.linalg.sqrtm(cov)
            tmpFeatures = np.array(
                [m1, m2, m3, srcov[0, 0], srcov[0, 1], srcov[0, 2], srcov[1, 1], srcov[1, 2], srcov[2, 2]])
            streamFeatures = np.vstack((streamFeatures, tmpFeatures))

        else:
            print("warning: np.linalg.det(cov)=0")
        lastindex = allidx[i]
    if len(streamFeatures.shape) == 2:
        return streamFeatures
    elif len(streamFeatures.shape) == 1:
        return streamFeatures.reshape(1, -1)
    
def constructCluster(ply, idx, streamFeatures, n_clusters=30):
    print("constructCluster")
    '''if streamFeatures == None:
        streamFeatures = covMatrixPly(ply,idx)'''
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(streamFeatures)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    selectedStreamIndex = []
    for i in k_means_labels_unique:
        dist = streamFeatures - k_means_cluster_centers[i]
        dist = dist[:] ** 2.0
        dist = dist.sum(axis=1)
        dist = np.sqrt(dist)
        selectedStreamIndex.append(dist.argmin())
    print("creating new bundle according to k-means . . .")
    # selectedStreamIndex.sort()

    mask = np.zeros(idx.shape, dtype=int)
    mask[selectedStreamIndex] = 1

    nidx = np.zeros(idx.shape, dtype=int)
    nidx[1:] = idx[:-1]
    didx = idx - nidx
    assert len(mask[mask == 1]) == n_clusters
    selectedStream = np.empty([0, 3])
    plyidx = []
    for i in np.where(mask == 1)[0]:
        plyidx.append(didx[i])
        selectedStream = np.vstack((selectedStream, ply[nidx[i]:idx[i], :]))
    plyidx = np.cumsum(plyidx)
    return selectedStream, np.asarray(plyidx)


def outlierSVM(features, nuv):
    clf = svm.OneClassSVM(kernel="rbf", nu=nuv)
    clf.fit(features)
    mask = clf.predict(features)
    newCov = np.cov(features[mask[:] == 1].T)
    Fnorm = np.sqrt(np.sum(newCov[:] ** 2))
    numFiber = mask[mask[:] == 1].shape[0]
    return Fnorm, numFiber, mask

def GPUoutlierSVM(features, nuv):
    features = cp.asarray(features)  # Convert to GPU array
    clf = cuml.svm.OneClassSVM(kernel="rbf", nu=nuv)
    clf.fit(features)
    mask = clf.predict(features)
    newCov = cp.cov(features[mask[:] == 1].T)
    Fnorm = cp.sqrt(cp.sum(newCov[:] ** 2))
    numFiber = mask[mask[:] == 1].shape[0]
    return Fnorm.get(), numFiber.get(), mask.get()  # Convert back to CPU array


def matFromFile(lmatPath):
    with open(lmatPath) as f:
        larray = []
        for line in f:
            larray.append([np.double(xval) for xval in line.split()])
    fileMat = np.asarray(larray)
    return fileMat


def niftiItoF(niftiImg):
    # inaffine=inv(niftiImg.get_affine())

    niftiZoom = niftiImg.get_header().get_zooms()
    ItoF = np.array([[niftiZoom[0], 0, 0, 0], [0, niftiZoom[1], 0, 0], [0, 0, niftiZoom[2], 0], [0, 0, 0, 1.0]])
    qform = niftiImg.get_qform()
    qform = qform[0:3, 0:3]
    detqform = np.linalg.det(qform)
    niftiShape = niftiImg.shape
    if (detqform > 0):
        ItoF[0, 0] = -niftiZoom[0]
        ItoF[0, 3] = niftiZoom[0] * (niftiShape[0] - 1)
    return ItoF


def rigid_transform_3DN(A, B):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points

    AA = A - np.mean(A, axis=0)
    BB = B - np.mean(B, axis=0)
    # print AA.shape
    # print BB.shape
    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)

    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A.T) + centroid_B.T

    return R, t


def Interpolation3DV(indexposf, vol3d):
    indexposf = indexposf.T
    indexposf[indexposf < 0] = 0
    indexposf[indexposf[:, 0] > vol3d.shape[0] - 1, 0] = vol3d.shape[0] - 1
    indexposf[indexposf[:, 1] > vol3d.shape[1] - 1, 1] = vol3d.shape[1] - 1
    indexposf[indexposf[:, 2] > vol3d.shape[2] - 1, 2] = vol3d.shape[2] - 1
    indexpos = indexposf.astype(int)

    scalarArray = np.zeros((indexpos.shape[0], 8))

    scalarArray[:, 0] = vol3d[indexpos[:, 0], indexpos[:, 1], indexpos[:, 2]]

    scalarArray[:, 1] = vol3d[np.minimum(indexpos[:, 0] + 1, vol3d.shape[0] - 1), indexpos[:, 1], indexpos[:, 2]]

    scalarArray[:, 2] = vol3d[indexpos[:, 0], np.minimum(indexpos[:, 1] + 1, vol3d.shape[1] - 1), indexpos[:, 2]]

    scalarArray[:, 3] = vol3d[indexpos[:, 0], indexpos[:, 1], np.minimum(indexpos[:, 2] + 1, vol3d.shape[2] - 1)]

    scalarArray[:, 4] = vol3d[
        np.minimum(indexpos[:, 0] + 1, vol3d.shape[0] - 1), indexpos[:, 1], np.minimum(indexpos[:, 2] + 1,
                                                                                       vol3d.shape[2] - 1)]

    scalarArray[:, 5] = vol3d[
        indexpos[:, 0], np.minimum(indexpos[:, 1] + 1, vol3d.shape[1] - 1), np.minimum(indexpos[:, 2] + 1,
                                                                                       vol3d.shape[2] - 1)]

    scalarArray[:, 6] = vol3d[np.minimum(indexpos[:, 0] + 1, vol3d.shape[0] - 1), np.minimum(indexpos[:, 1] + 1,
                                                                                             vol3d.shape[
                                                                                                 1] - 1), indexpos[:,
                                                                                                          2]]

    scalarArray[:, 7] = vol3d[np.minimum(indexpos[:, 0] + 1, vol3d.shape[0] - 1), np.minimum(indexpos[:, 1] + 1,
                                                                                             vol3d.shape[
                                                                                                 1] - 1), np.minimum(
        indexpos[:, 2] + 1, vol3d.shape[2] - 1)]

    xd = indexposf[:, 0] - indexpos[:, 0]
    yd = indexposf[:, 1] - indexpos[:, 1]
    zd = indexposf[:, 2] - indexpos[:, 2]
    # print fieldArray.shape
    # pickle.dump(fieldArray, open( "/home/khatami/script/fieldArray.p" , "wb" ))

    scalarVal = scalarArray[:, 0].T * ((1 - xd) * (1 - yd) * (1 - zd))
    scalarVal += scalarArray[:, 1].T * xd * (1 - yd) * (1 - zd)
    scalarVal += scalarArray[:, 2].T * (1 - xd) * yd * (1 - zd)
    scalarVal += scalarArray[:, 3].T * (1 - xd) * (1 - yd) * zd
    scalarVal += scalarArray[:, 4].T * xd * (1 - yd) * zd
    scalarVal += scalarArray[:, 5].T * (1 - xd) * yd * zd
    scalarVal += scalarArray[:, 6].T * (xd * yd * (1 - zd))
    scalarVal += scalarArray[:, 7].T * (xd * yd * zd)

    return scalarVal


def InterpolationField(indexposf, field):
    indexposf = indexposf.T
    indexposf[indexposf < 0] = 0
    indexposf[indexposf[:, 0] > field.shape[0] - 1, 0] = field.shape[0] - 1
    indexposf[indexposf[:, 1] > field.shape[1] - 1, 1] = field.shape[1] - 1
    indexposf[indexposf[:, 2] > field.shape[2] - 1, 2] = field.shape[2] - 1
    indexpos = indexposf.astype(int)

    fieldArray = np.zeros((indexpos.shape[0], 3, 8))

    fieldArray[:, :, 0] = field[indexpos[:, 0], indexpos[:, 1], indexpos[:, 2], :]

    fieldArray[:, :, 1] = field[np.minimum(indexpos[:, 0] + 1, field.shape[0] - 1), indexpos[:, 1], indexpos[:, 2], :]

    fieldArray[:, :, 2] = field[indexpos[:, 0], np.minimum(indexpos[:, 1] + 1, field.shape[1] - 1), indexpos[:, 2], :]

    fieldArray[:, :, 3] = field[indexpos[:, 0], indexpos[:, 1], np.minimum(indexpos[:, 2] + 1, field.shape[2] - 1), :]

    fieldArray[:, :, 4] = field[np.minimum(indexpos[:, 0] + 1, field.shape[0] - 1), indexpos[:, 1],
                          np.minimum(indexpos[:, 2] + 1, field.shape[2] - 1), :]

    fieldArray[:, :, 5] = field[indexpos[:, 0], np.minimum(indexpos[:, 1] + 1, field.shape[1] - 1),
                          np.minimum(indexpos[:, 2] + 1, field.shape[2] - 1), :]

    fieldArray[:, :, 6] = field[np.minimum(indexpos[:, 0] + 1, field.shape[0] - 1),
                          np.minimum(indexpos[:, 1] + 1, field.shape[1] - 1), indexpos[:, 2], :]

    fieldArray[:, :, 7] = field[np.minimum(indexpos[:, 0] + 1, field.shape[0] - 1),
                          np.minimum(indexpos[:, 1] + 1, field.shape[1] - 1),
                          np.minimum(indexpos[:, 2] + 1, field.shape[2] - 1), :]

    xd = indexposf[:, 0] - indexpos[:, 0]
    yd = indexposf[:, 1] - indexpos[:, 1]
    zd = indexposf[:, 2] - indexpos[:, 2]
    # print fieldArray.shape
    # pickle.dump(fieldArray, open( "/home/khatami/script/fieldArray.p" , "wb" ))

    vfield = (fieldArray[:, :, 0].T * ((1 - xd) * (1 - yd) * (1 - zd))).T
    vfield += (fieldArray[:, :, 1].T * xd * (1 - yd) * (1 - zd)).T
    vfield += (fieldArray[:, :, 2].T * (1 - xd) * yd * (1 - zd)).T
    vfield += (fieldArray[:, :, 3].T * (1 - xd) * (1 - yd) * zd).T
    vfield += (fieldArray[:, :, 4].T * xd * (1 - yd) * zd).T
    vfield += (fieldArray[:, :, 5].T * (1 - xd) * yd * zd).T
    vfield += (fieldArray[:, :, 6].T * (xd * yd * (1 - zd))).T
    vfield += (fieldArray[:, :, 7].T * (xd * yd * zd)).T
    return vfield
