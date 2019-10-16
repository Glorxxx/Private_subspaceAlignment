import numpy as np
import scipy.io
import scipy.linalg
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def Z_Score(X):
    #axis=1 calculate by column
    X-=np.mean(X,axis=0)
    X/=np.std(X,axis=0)

    return X

def noisyCount(sensitivety,epsilon):
    beta = sensitivety/epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta*np.log(1.-u2)
    else:
        n_value = beta*np.log(u2)
    #print(n_value)
    return n_value

def laplace_mech(data,sensitivety,epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivety,epsilon)
    return data

# used for PCA 零均值化
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal


def PCA(dataMat):
    # used for PCA 协方差矩阵
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)
    #根据协方差矩阵求特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    #保留主要成分
    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(81):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    return n_eigVect
def PCA_noisy(datamat):
    newData,meanVal=zeroMean(datamat)
    covMat=np.cov(newData,rowvar=0)
    sensitivity = 2 * datamat.shape[1] / datamat.shape[0]
    covMat= laplace_mech(covMat,sensitivity,1)
    # 根据协方差矩阵求特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 保留主要成分
    eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
    n_eigValIndice = eigValIndice[-1:-(81):-1]  # 最大的n个特征值的下标
    n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
    return n_eigVect

def generateTrainSplits(original_train_lbl,original_trainset,num_trn_lbls):
    classes=np.unique(original_train_lbl)
    im_index=[]
    for i in range(0,len(classes)+1):
        count=0
        for j in range(0,len(original_train_lbl)):
            if original_train_lbl[j][0]==i:
                count=count+1
                im_index.append(j)
            if count>19:
                break
    original_train_lbl = original_train_lbl[im_index, :]
    original_trainset = original_trainset[im_index, :]
    return  original_train_lbl,original_trainset

def subspace_Alignment(Source_Data,Target_Data,Xs,Xt):
    Source_Data=np.mat(Source_Data)
    Target_Aligned_Source_Data = Source_Data*(Xs * Xs.T*Xt)
    Target_Projected_Data = Target_Data * Xt
    return Target_Aligned_Source_Data,Target_Projected_Data

def PCA_depo_noise(dataMat):
    newData, meanval = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)
    sensitivity = 2 * dataMat.shape[1] / dataMat.shape[0]
    epsilon=1
    covMat = laplace_mech(covMat, sensitivity, epsilon)
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigvals)
    n_eigValIndice = eigValIndice[-1:-(81):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    #打乱顺序
    n_eigVect=np.array(n_eigVect)
    n_eigVect = n_eigVect.T[np.lexsort(n_eigVect)].T
    n_eigVect=np.mat(n_eigVect)
    #换个分的方式 前加后 中间的放一起,加overlap
    list=[]
    a=np.hsplit(n_eigVect, 4)
    l1 = np.hstack((a[0], a[1]))
    l1=np.hstack((l1,a[2]))
    l2 = np.hstack((a[1], a[2]))
    l2= np.hstack((l2,a[3]))
    list.append(l1)
    list.append(l2)
    #return np.hsplit(n_eigVect, 2)
    return list
def PCA_depo(dataMat):
    newData, meanval = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)
    eigvals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigvals)
    n_eigValIndice = eigValIndice[-1:-(81):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    # 打乱顺序
    n_eigVect=np.array(n_eigVect)
    n_eigVect = n_eigVect.T[np.lexsort(n_eigVect)].T
    n_eigVect=np.mat(n_eigVect)
    # 换个分的方式 前加后 中间的放一起
    list=[]
    a=np.hsplit(n_eigVect, 4)
    l1 = np.hstack((a[0], a[1]))
    l1=np.hstack((l1,a[2]))
    l2 = np.hstack((a[1], a[2]))
    l2= np.hstack((l2,a[3]))
    list.append(l1)
    list.append(l2)
    #return np.hsplit(n_eigVect, 2)
    return list
if __name__ == '__main__':
    #load data and preprocessing
    src, tar = 'data/amazon_SURF_L10.mat' , 'data/Caltech10_SURF_L10.mat'
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    S, Ys  = src_domain['fts'], src_domain['labels']
    T, Yt =  tar_domain['fts'], tar_domain['labels']
    S=np.mat(S)
    S=S/np.tile(S.sum(axis=1),np.shape(S)[1])
    S=preprocessing.scale(S)
    #S=np.array(S)
    T = np.mat(T)
    T = T/ np.tile(T.sum(axis=1), np.shape(T)[1])
    T = preprocessing.scale(T)

    T=np.array(T)
    S = np.array(S)
    #PCA
    Xs = PCA(S)
    Xt = PCA(T)
    #calculate sensitivity
    # S1=S[0:956,:]
    # A=np.cov(S,rowvar=0)
    # A1 = np.cov(S1, rowvar=0)
    # a=A-A1
    # sensitivity=np.linalg.norm(a, ord=1)
    sensitivity=2*S.shape[1]/S.shape[0]
    #generate splits source data
    #Ys_new,S_new=generateTrainSplits(Ys,S,20)
    Ys_new, S_new =Ys,S
    #subspaceAlignment
    Target_Aligned_Source_Data,Target_Projected_Data=subspace_Alignment(S_new,T,Xs,Xt)
    # train a clssifier with subspace alignment
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(Target_Aligned_Source_Data, Ys_new.ravel())
    y_pred_1 = clf.predict(Target_Projected_Data)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred_1)
    print(acc)
    # without subspace alignment
    clf.fit(S_new, Ys_new.ravel())
    y_pred_2 = clf.predict(T)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred_2)
    print(acc)

    # add noise to covariance matrix
    Xs = PCA(S)
    Xt = PCA_noisy(T).real
    Target_Aligned_Source_Data,Target_Projected_Data=subspace_Alignment(S_new,T,Xs,Xt)
    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(Target_Aligned_Source_Data, Ys_new.ravel())
    y_pred_3 = clf.predict(Target_Projected_Data)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred_3)
    print(acc)

    #特征分割
    Xs = PCA_depo(S)
    Xt = PCA_depo_noise(T)
    #各个模型投票选出最终结果
    acc_list=[]
    for i in range(0, len(Xs)):
        Target_Aligned_Source_Data = (S_new*Xs[i] * Xs[i].T * Xt[i]).real
        for j in range(0,len(Xs)):
            Target_Projected_Data = T*Xt[j].real
            clf = KNeighborsClassifier(n_neighbors=4)
            clf.fit(Target_Aligned_Source_Data, Ys_new.ravel())
            y_pred_4 = clf.predict(Target_Projected_Data)
            acc = sklearn.metrics.accuracy_score(Yt, y_pred_4)
            acc_list.append(acc)
            y_pred_3=np.vstack((y_pred_3,y_pred_4))
    y_pred=y_pred_3[1:len(y_pred_3)-1,:]
    #当预测结果的数量相同时
    list=[]
    for i in range(0,y_pred.shape[1]):
        a=Counter(y_pred[:,i])
        if a.most_common(1)[0][1]==a.most_common(2)[0][1]:#1是个数，0是对应的元素
            bb=[k for k in range(len(y_pred[:,i])) if a[k]== a.most_common(1)[0][0]]
            accuracy1=0
            for f in range(len(bb)):
                if acc_list[bb[f]]> accuracy1:
                    accuracy1=acc_list[bb[f]]
            bb = [k for k in range(len(y_pred[:, i])) if a[k] == a.most_common(2)[0][0]]
            accuracy2 = 0
            for j in range(len(bb)):
                if acc_list[bb[j]] > accuracy2:
                    accuracy1 = acc_list[bb[j]]
            if accuracy1 > accuracy2 :
                result=a.most_common(1)[0][0]
            else:
                result = a.most_common(2)[0][0]
        else:
            result=a.most_common(1)[0][0]
        list.append(result)
    y_pred=np.array(list)
    acc = sklearn.metrics.accuracy_score(Yt, y_pred)
    print(acc)
    print(acc_list)
    #print(1)

    # #add noise to target pcaxt(need to calculate the sensitivity)
    # sensitivety = 1
    # epsilon = 1
    # Xt_noise = laplace_mech(Xt,sensitivety,epsilon)
    #
    # #transform2
    # Xa=np.dot(Xs,Xs.T)
    # Xa=np.dot(Xa,Xt)
    # Sa=np.dot(S1,Xa)
    # Tt=np.dot(T,Xt)
    #








