from scipy.stats import wasserstein_distance
import pandas as pd
from sklearn.metrics import mutual_info_score,accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate
from sklearn.svm import SVC
from sklearn import feature_selection
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skfeature.utility.sparse_learning import construct_label_matrix,feature_ranking
from skfeature.function.information_theoretical_based import MRMR,CIFE,CMIM,DISR,ICAP,JMI,MIFS,MIM
from skfeature.function.similarity_based import reliefF
from skfeature.function.statistical_based import CFS
from skfeature.function.sparse_learning_based import RFS
import time

# data = pd.read_csv('chess_zscore.csv')
# data = pd.read_csv('chess.csv',sep='\t')
# data = pd.read_csv('arrhythmia.csv')
# data = pd.read_csv('breast_cancer.csv',sep='\t')
# data = pd.read_csv('breast_tissue_zscore.csv',sep='\t')
data = pd.read_csv('crx_zscore.csv')
# data = pd.read_csv('messidor_zscore.csv')
# data = pd.read_csv('EEG_eye_zscore.csv')
# data = pd.read_csv('Eplieptic_zscore.csv')
# data = pd.read_csv('Z-Alizadeh_zscore.csv')
# data = pd.read_csv('iBeacon_RSSI_zscore.csv')
# data = pd.read_csv('breast_tissue_zscore.csv',sep='\t')
# data = pd.read_csv('krkopt_zscore.csv',sep='\t')
# data = pd.read_csv('sonar_zscore.csv')
# data = pd.read_csv('vowel_zscore.csv')
# data = pd.read_csv('LSVT_zscore.csv')
# data = pd.read_csv('ionosphere_zscore.csv')
X = data
y = data['label']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
for trian_index,test_index in sss.split(X,y):
    X_test,X_train = X.loc[trian_index],X.loc[test_index]
    y_test,y_train = y.loc[trian_index],y.loc[test_index]

column = X.columns
maxcolumn_num = len(column) - 1 #remove 'label' column

y_label = list(set(y))
num_class = len(y_label)
X_train_class = []
for c_label in y_label:
    X_train_class.append(X_train[X_train['label']==c_label])

# create MI matrix
total_num = (len(column)-1)*(len(column)-1)
#对角线是相同的特征，那么可以用来表示同一特征正负之间的EMD，不同特征之间则可以用MI表示
emd_mi_matrix = pd.DataFrame(np.arange(total_num).reshape((len(column)-1,len(column)-1)),index=column[:-1],columns=column[:-1],dtype=float)
# print(emd_mi_matrix)
for i in range(maxcolumn_num-1):
    v1 = column[i]
    for j in range(i+1,maxcolumn_num):
        v2 = column[j]
        mi = mutual_info_score(X_train[v1],X_train[v2])
        # print(mi)
        emd_mi_matrix[v1][v2] = emd_mi_matrix[v2][v1] = mi

for i in range(maxcolumn_num):
    v = column[i]
    emd = 0
    # emd = wasserstein_distance(X_train0[v],X_train1[v])
    for j in range(num_class - 1):
        for k in range(j + 1, num_class):
            emd += wasserstein_distance(X_train_class[j][v], X_train_class[k][v])
    emd_mi_matrix[v][v] = emd

MI = []
for i in range(0,maxcolumn_num-1):
    v1 = column[i]
    for j in range(i+1,maxcolumn_num):
        v2 = column[j]
        MI.append(emd_mi_matrix[v1][v2])
ave_MI = sum(MI)/(maxcolumn_num*maxcolumn_num)

EMD = []
for i in range(maxcolumn_num):
    v1 = column[i]
    emd = emd_mi_matrix[v1][v1]
    EMD.append(emd)
    # print(v1,' ',emd)
ave_emd = sum(EMD)/maxcolumn_num

alpha = ave_MI/(ave_emd+ave_MI)

#column[-1]='label'

max_j = float('-inf') #object function value max_js

for i in range(maxcolumn_num-1):
    v1 = column[i]
    for j in range(i+1,maxcolumn_num):
        v2 = column[j]
        emd = emd_mi_matrix[v1][v1] + emd_mi_matrix[v2][v2]
        mi = emd_mi_matrix[v1][v2]
        js = alpha*0.5*emd - (1-alpha)*0.25*mi
        # print(v1,v2)
        # print('emd',emd)
        # print('mi',mi)
        if js >= max_j:
            max_j = js
            max_v1 = v1
            max_v2 = v2

selected_feature = [max_v1,max_v2]

def com_acc(selected_feature):
    clf = SVC()
    X_train_new = pd.DataFrame(X,columns=selected_feature)
    scores = cross_validate(clf,X_train_new,y,cv=10)
    accs = np.asarray(scores['test_score'])
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    return [acc_mean,acc_std]


def com_mi(selected_feature,new_feature):
    mi = 0
    for i in range(len(selected_feature)-1):
        v1 = selected_feature[i]
        for j in range(i+1,len(selected_feature)):
            v2 = selected_feature[j]
            mi += emd_mi_matrix[v1][v2]
        mi += emd_mi_matrix[v1][new_feature]
    mi += emd_mi_matrix[selected_feature[-1]][new_feature]

    return mi

def com_emd(selected_feature,new_feature):
    emd = 0
    for i in range(len(selected_feature)):
        vi = selected_feature[i]
        emd += emd_mi_matrix[vi][vi]
    emd += emd_mi_matrix[new_feature][new_feature]

    return emd
acc_mean,acc_std = com_acc(selected_feature)
print(len(selected_feature),max_j,acc_mean,acc_std,sep=' ')
for n in range(maxcolumn_num-2): #select 3,4,...,maxcolumn features
    max_j = float('-inf')
    s = len(selected_feature) + 1
    for i in range(maxcolumn_num):
        vi = column[i]
        if vi in selected_feature:
            continue
        # j = alpha/s*com_emd(selected_feature,vi) - (1-alpha)/(s*s)*(com_mi(selected_feature,vi))
        j = 1 / s * com_emd(selected_feature, vi) - 1 / (s * s) * (com_mi(selected_feature, vi))

        if j >= max_j:
            max_j = j
            selecte_f = vi
    selected_feature.append(selecte_f)
    acc_mean,acc_std = com_acc(selected_feature)
    print(len(selected_feature),  max_j, acc_mean, acc_std,sep=' ')



X = X.drop(['label'],axis=1)
#MRMR-----------------------------------------------------------------------
print('MRMR')
X_new = X.values
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
for i in range(2,maxcolumn_num+1):
    idx,_,_ = MRMR.mrmr(X_new,y,n_selected_features=i)
    X_train = X_new[:,idx[0:i]]
    clf = SVC()
    scores = cross_validate(clf, X_train, y, cv=10)
    accs = np.asarray(scores['test_score'])
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)

    print(acc_mean,acc_std,sep=' ')


#reliefF--------------------------------------------------------------------
print('relifF')
X_new = X.values
score = reliefF.reliefF(X_new, y)
idx = reliefF.feature_ranking(score)
for i in range(2,maxcolumn_num+1):
    X_train = X_new[:, idx[0:i]]
    clf = SVC()
    scores = cross_validate(clf, X_train, y, cv=10)
    accs = np.asarray(scores['test_score'])
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)

    print(acc_mean, acc_std, sep=' ')

#
# #CIFE-----------------------------------------------------------------------
# print('cife')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx, _, _ = CIFE.cife(X_new, y, n_selected_features=i)
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
# #CMIM-----------------------------------------------------------------------
# print('cmim')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx, _, _ = CMIM.cmim(X_new, y, n_selected_features=i)
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
# #DISR-----------------------------------------------------------------------
# print('disr')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx, _, _ = DISR.disr(X_new, y, n_selected_features=i)
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
# #ICAP-----------------------------------------------------------------------
# print('icap')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx, _, _ = ICAP.icap(X_new, y, n_selected_features=i)
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
#
# #JMI-----------------------------------------------------------------------
# print('jmi')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx, _, _ = JMI.jmi(X_new, y, n_selected_features=i)
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
#
# #MIFS-----------------------------------------------------------------------
# print('mifs')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx = MIFS.mifs(X_new, y, n_selected_features=i)
#     idx = list(idx[0])
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
#
# # MIM-----------------------------------------------------------------------
# print('mim')
# X_new = X.values
# for i in range(2,maxcolumn_num+1):
#     idx, _, _ = MIM.mim(X_new, y, n_selected_features=i)
#     X_train = X_new[:, idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
#
# #RFS-----------------------------------------------------------------------
# print('rfs')
# X_new = X.values
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
#
# for trian_index,test_index in sss.split(X,y):
#     X_train,y_train = X_new[test_index],y[test_index]
#     X_test,y_test = X_new[trian_index],y[trian_index]
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# def to_onehot(ls):
#     one_hot = []
#     for i in ls:
#         if i==0:
#             one_hot.append([1,0])
#         else:
#             one_hot.append([0,1])
#     return one_hot
#
# y_train1 = to_onehot(y_train)
#
# weight = RFS.rfs(X_train,y_train1,gamma=0.1)
# idx = feature_ranking(weight)
#
# for i in range(2,len(idx)+1):
#     X_train = X_new[:,idx[0:i]]
#     clf = SVC()
#     scores = cross_validate(clf, X_train, y, cv=10)
#     accs = np.asarray(scores['test_score'])
#     acc_mean = np.mean(accs)
#     acc_std = np.std(accs)
#
#     print(acc_mean, acc_std, sep=' ')
#
#
#
# #cfs------------------------------------------------
# print('cfs')
# X_new = X.values
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
#
# for trian_index,test_index in sss.split(X,y):
#     X_train,y_train = X_new[test_index],y[test_index]
#     X_test,y_test = X_new[trian_index],y[trian_index]
# y_train = np.array(y_train)
#
# idx = CFS.cfs(X_train,y_train)
# print('[',len(idx),']')
#
# X_train = X_new[:,idx[0:-1]]
# clf = SVC()
# scores = cross_validate(clf, X_train, y, cv=10)
# accs = np.asarray(scores['test_score'])
# acc_mean = np.mean(accs)
# acc_std = np.std(accs)
#
# print(acc_mean, acc_std, sep=' ')






