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
from math import log

# data = pd.read_csv('chess_zscore.csv')
# data = pd.read_csv('chess.csv',sep='\t')
# data = pd.read_csv('arrhythmia.csv')
# data = pd.read_csv('breast_cancer.csv',sep='\t')
# data = pd.read_csv('breast_tissue_zscore.csv',sep='\t')
# data = pd.read_csv('crx_zscore.csv')
# data = pd.read_csv('messidor_zscore.csv')
# data = pd.read_csv('EEG_eye_zscore.csv')
# data = pd.read_csv('Eplieptic_zscore.csv')#暂未处理
# data = pd.read_csv('Z-Alizadeh_zscore.csv')
# data = pd.read_csv('iBeacon_RSSI_zscore.csv')
# data = pd.read_csv('krkopt_zscore.csv',sep='\t')#暂未处理
data = pd.read_csv('sonar_zscore.csv')
# data = pd.read_csv('vowel_zscore.csv')
# data = pd.read_csv('LSVT_zscore.csv')#暂未处理
# data = pd.read_csv('ionosphere_zscore.csv')
X = data
y = data['label']

#取得字典最大值的键
def get_keys(d,value):
    return [k for k,v in d.items() if v==value]


def calc_ent(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

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

#create shannonent
H={}
for i in range(maxcolumn_num):
    v = column[i]
    H[v]=calc_ent(X_train[[v,'label']])

# create MI matrix
total_num = (len(column)-1)*(len(column)-1)
#MI表示
emd_mi_matrix = pd.DataFrame(np.arange(total_num).reshape((len(column)-1,len(column)-1)),index=column[:-1],columns=column[:-1],dtype=float)
# print(emd_mi_matrix)
for i in range(maxcolumn_num-1):
    v1 = column[i]
    for j in range(i+1,maxcolumn_num):
        v2 = column[j]
        mi = mutual_info_score(X_train[v1],X_train[v2])
        # print(mi)
        emd_mi_matrix[v1][v2] = emd_mi_matrix[v2][v1] = mi


max_j = float('-inf') #object function value max_js

#计算I(f_i;C)每个特征自己的互信息
I={}
for i in range(maxcolumn_num):
    v = column[i]
    I[v] = mutual_info_score(X_train[v],X_train['label'])

first_feature = get_keys(I,max(I.values()))[0]

selected_feature = []
selected_feature.append(first_feature)

def com_acc(selected_feature):
    clf = SVC()
    X_train_new = pd.DataFrame(X,columns=selected_feature)
    scores = cross_validate(clf,X_train_new,y,cv=10)
    accs = np.asarray(scores['test_score'])
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    return [acc_mean,acc_std]

acc_mean,acc_std = com_acc(selected_feature)
print(len(selected_feature),acc_mean,acc_std,sep=' ')


def com_NI(f1,f2):
    mutual_i = emd_mi_matrix[f1][f2]
    min_h = min(H[f1],H[f2])
    return mutual_i/min_h

#G公式的右边
def right(f1,select):
    len_s = len(select)
    ni = 0
    for v in select:
        ni += com_NI(f1,v)

    return ni/len_s

column2 = list(X.columns)
column2.remove('label')
column2.remove(selected_feature[0])

def get_g():
    max_g = float('-inf')
    selected = None
    for i in range(len(column2)):
        g = I[column2[i]]-right(column2[i],selected_feature)
        if g>max_g:
            max_g = g
            selected = column2[i]
    return selected

for i in range(len(column2)):
    selected_feature.append(get_g())
    # print(selected_feature)
    acc_mean,acc_std = com_acc(selected_feature)
    print(len(selected_feature),acc_mean, acc_std, sep=' ')