#!/usr/bin/env python
# encoding:utf-8

'''sklearn doc
'''

import re  
import os
import sys
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import GridSearchCV, cross_val_predict  

# RandomizedSearchCV cross_val_score train_test_split


from skfeature.function.information_theoretical_based import MRMR
from imblearn.over_sampling import SMOTE
# from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression,f_classif
# from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from joblib import Memory, dump, load
from sklearn import metrics
from pycm import *   #swiss-army knife of confusion matrice
from collections import Counter

# from sklearn.base import BaseEstimator,TransformerMixin
# from imblearn.metrics import classification_report_imbalanced

import utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') #UserWarning: 
from plotnine import *  #ggplot

#Global variables
mem = Memory("./mycache") #A context object for caching a function's return value each time it is called with the same input arguments.

import itertools

# COLORS = 'bgrcmyk' #blue green red itertools.cycle(cmap.colors))
# cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
#                         'Dark2', 'Set1', 'Set2', 'Set3',
#                         'tab10', 'tab20', 'tab20b', 'tab20c']

cmap = plt.get_cmap('Paired')
COLORS = cmap.colors

from sklearn_pipeline_config import * #SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS RANDOM_STATE 
All_CLASSIFIERS =  Tree_based_CLASSIFIERS + Other_CLASSIFIERS
    
######################## pipeline functions  ###################
def plot_tsne(df, Y=None, targets=None, filename='decomposition'):
    """to be fihished
    method= ['tsne', 'pca', 'tsvd']
    t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results
    PCA for dense data or 
    TruncatedSVD for sparse data
    但TSVD直接使用scipy.sparse矩阵，不需要densify操作，所以推荐使用TSVD而不是PCA
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA, TruncatedSVD

    n_components = min(df.shape) if min(df.shape) <10 else 10
    X = TSNE(random_state=RANDOM_STATE, learning_rate=100, n_components=2).fit_transform(df) 
    pd.DataFrame(X).to_csv(filename + ".tSNE.csv")
    
    fig = plt.figure(figsize=(10, 6))
    for c, i, target_name in zip('rgb', np.unique(Y),  targets):
        plt.scatter(X[Y==i, 0], X[Y==i, 1], c=c,  label=target_name)
    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')
    plt.title('tSNE')
    plt.legend()
    fig.savefig(filename + ".tSNE.svg")
    
    #pca
    pca = PCA(random_state=RANDOM_STATE, n_components=n_components)
    pca.fit(df)
    X = pca.transform(df) 
    pd.DataFrame(X).to_csv(filename + ".pca.csv")
    
    fig = plt.figure(figsize=(10, 6))
    for c, i, target_name in zip('rgb', np.unique(Y),  targets):
        plt.scatter(X[Y==i, 0], X[Y==i, 1], c=c,  label=target_name)
    p1,p2=pca.explained_variance_ratio_[:2]
    plt.xlabel('PCA-1 explained variance ratio: ' + '{:.2f}%'.format(p1))
    plt.ylabel('PCA-2 explained variance ratio: ' + '{:.2f}%'.format(p2))
    plt.title('PCA')
    plt.legend()
#     print("singular_values: ", pca.singular_values_)  
    fig.savefig(filename + ".pca.svg")
    
    #tSVD
    tsvd=TruncatedSVD(random_state=RANDOM_STATE, n_components=n_components) 
    tsvd.fit(df)
    X = tsvd.transform(df)
    pd.DataFrame(X).to_csv(filename + ".tSVD.csv")
    
    fig = plt.figure(figsize=(10, 6))
    for c, i, target_name in zip('rgb', np.unique(Y),  targets):
        plt.scatter(X[Y==i, 0], X[Y==i, 1], c=c,  label=target_name)
    p1,p2=tsvd.explained_variance_ratio_[:2]
    plt.xlabel('tSVD-1 explained variance ratio: ' + '{:.2f}%'.format(p1))
    plt.ylabel('tSVD-2 explained variance ratio: ' + '{:.2f}%'.format(p1))
    plt.title('tSVD')
    plt.legend()
    fig.savefig(filename + ".tSVD.svg")

    
@mem.cache
def get_data(X_file, y_file):
    """features matrix and metadata group.mf with header and index_col,transform to relative abundance matrix"""
    if X_file.endswith("csv"):
        X = pd.read_csv(X_file, index_col=0, header=0)  # rows =samples ,columns=genes(features)
    else:
        X = pd.read_csv(X_file, index_col=0, header=0,sep="\t")
    if y_file.endswith("csv"):
        y = pd.read_csv(y_file, index_col=0, header=0)  # rows =samples
    else:
        y = pd.read_csv(y_file, index_col=0, header=0,sep="\t")
    return X, y


def plot_classification_report(dict_report, filename="sklearn",
                          width=6, heightight=3,dpi=300):
    report_df = round(pd.DataFrame(dict_report), 2) #保留2位小数
    report_df.to_csv(filename + ".classification_report.csv") 

    report_df = report_df.loc[report_df.index != 'support',]
    report_df.insert(0,'score',report_df.index)
    plt_df = report_df.melt(id_vars=['score'], value_vars=report_df.columns[1:])
    base_plot=(ggplot(plt_df, aes( y = plt_df.columns[1],x=plt_df.columns[-1])) + 
               geom_point(aes(fill="factor(variable)"),stat='identity',show_legend=False)+
               facet_grid('~score')+  #,scales="free_x"
               xlim(0,1)+
               theme_bw()+
               labs(x="",y="")
              )
    base_plot.save(filename=filename + '.classification_report.svg', dpi=dpi,width=width, height=heightight)  

    
def report_topN_cv_results(results, n_top=10, filename="report"):
    """输出topn，评估那个标准化和分类器最好，用gridsearch"""
    labels = []
    mean_train_score=[]
    mean_test_score=[]
    std_test_score=[]
    mean_fit_time=[]
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            labeldict={key:value.__class__.__name__ for key, value in results['params'][candidate].items()}
            label = "_".join(labeldict[k] for k in ["scl","clf"])
            labels.append(label)
            mean_train_score.append(results['mean_train_score'][candidate])
            mean_test_score.append(results['mean_test_score'][candidate])
            std_test_score.append(results['std_test_score'][candidate])
            mean_fit_time.append(results['mean_fit_time'][candidate])
    df = pd.DataFrame.from_dict(
        dict(zip(['label','mean_train_score', 'mean_test_score', 'std_test_score', 'mean_fit_time'],
                 (labels, mean_train_score, mean_test_score, std_test_score, mean_fit_time))))
    df.to_csv(filename + ".top{}.cv_results.csv".format(n_top) , index=False)
    
    fig = plt.figure(figsize=(12,5)) #fig size
#     plt.grid(which='major', axis='both')
    # You should use add_axes if you want exact control of the figure layout. eg.
    left = max([len(label) for label in labels])*0.008
    bottom, width, height=[0.2, 0.5, 0.7]
    ax =  fig.add_axes([left, bottom, width, height]) #axes position
    ax.barh(labels, mean_test_score, xerr=std_test_score, align='center', color=COLORS, ecolor='black')
    # ax.set_title("Compare the different scalers")
    ax.set_xlabel('Classification accuracy')
    # ax.set_ylabel('') #Different scalers
    ax.set_yticklabels(labels)
    ax.autoscale()
    fig.savefig(filename + ".top{}.cv_results.svg".format(n_top))
    

def csv2pycm_report(cm_csv):
    """readin cfm csv file and output report for multiple matrics"""
    df = pd.read_csv(cm_csv, index_col=0, header=0)
    matrix = df.T.to_dict()
    cm=ConfusionMatrix(matrix=matrix)
    cm.save_html(cm_csv + ".report")
    cm.save_csv(cm_csv + '.report')
    
    
def plot_confusion_matrix(cfm_df, filename="confusionmatrix", cmap=plt.cm.Blues, accuracy=None):
    """or plt.cm.gray"""
    cfm_df.to_csv(filename + ".csv")
    labels = list(cfm_df.columns)
    fig, ax = plt.subplots()
    fig.set_size_inches(8,8)
    cfm_df_norm = cfm_df.astype('float') / cfm_df.sum(axis=1)
#     cax = ax.matshow(cfm_df, cmap=cmap)
    ax.imshow(cfm_df, interpolation='nearest', cmap=cmap)
#     ax.set_title("Accuracy: " + accuracy) # plt.title('title test',fontsize=12,color='r')
    ax.xaxis.set_ticks_position('bottom')
    if isinstance(labels,list):
        ax.set(xticks=np.arange(cfm_df.shape[1]+1)-.5,
               yticks=np.arange(cfm_df.shape[0]+1)-.5,
               # ... and label them with the respective list entries
               yticklabels=labels,
               title="Accuracy: " + accuracy,
               ylabel='True label',
               xlabel='Predicted label')
    ax.tick_params(length=.0)
#     plt.xlabel('Predicted label')
#     plt.ylabel('True label')
#     ax.legend().set_visible(False) #no legend
    ax.set_xticklabels(labels, rotation=45)
    fmt = '.2f'
    thresh = 0.4 # max(cfm_df.max()) / 2.
    for i, j in itertools.product(range(cfm_df.shape[0]), range(cfm_df.shape[1])):
        ax.text(j, i+0.1, format(cfm_df_norm.iloc[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cfm_df_norm.iloc[i, j] > thresh else "black")
        ax.text(j, i-0.1, cfm_df.iloc[i, j],
                 horizontalalignment="center",
                 color="white" if cfm_df_norm.iloc[i, j] > thresh else "black")
    plt.tight_layout()    
    fig.savefig(filename + ".svg")
    
def plot_mean_test_scores(labels,mean_test_score,error,filename):
    """评估那个标准化和特征筛选最好，用gridsearch"""
    fig = plt.figure(figsize=(8,4)) #fig size
    plt.grid(which='major', axis='both')
    # You should use add_axes if you want exact control of the figure layout. eg.
    left = max([len(label) for label in labels])*0.01 +0.1
    bottom, width, height=[0.2, 0.5, 0.7]
    ax =  fig.add_axes([left, bottom, width, height]) #axes position
    ax.barh(labels, mean_test_score, xerr=error, align='center', color=COLORS, ecolor='black')
    # ax.set_title("Compare the different scalers")
    ax.set_xlabel('Classification accuracy')
    # ax.set_ylabel('') #Different scalers
    ax.set_yticklabels(labels)
    ax.autoscale()
    fig.savefig(filename)

def plot_feature_importance(df, k, filename):
    """plot feature importance for LGBMClassifier
    column0 features, column_1 importance
    """
    fig = plt.figure(figsize=(6,8))
#     plt.grid(which='major', axis='both')
    left = max([len(label) for label in df.iloc[:,0]])*0.01 +0.1
    bottom, width, height=[0.1, 1-left-0.1, 0.85]
    indices_of_top_k = np.sort(np.argpartition(np.array(df.iloc[:,1]), -k)[-k:])
    #np.argpartition: index, all smaller elements will be moved before it and all larger elements behind it
    #argpartition的效率比较高,据说是O(n) index 操作
    df = df.iloc[indices_of_top_k,].sort_values(by='importance')
    ax = fig.add_axes([left, bottom, width, height]) #axes position
    ax.barh(df.iloc[:,0],df.iloc[:,1])
    ax.set_ylim(-0.5,k-0.5)
    ax.set_xlim(0,max(df.importance)*1.1)
    for i, v in enumerate(df.iloc[:,1]):
        ax.text(v, i, '{0:0.2f}'.format(v), fontsize=8,
                horizontalalignment='left',verticalalignment='center') 
    ax.set_xlabel("Feature importance")
    fig.savefig(filename)
    top_k_feature_names = df.feature_names
    return top_k_feature_names


def plot_coefficients(df, topk=20, filename="filename"):
    """coefficients dataframe """
    # Access the classes
    df = df.reindex(df.abs().sum(axis=1).sort_values(ascending=False).index).head(topk) 
    classes = df.columns
    n_classes = len(classes)
    df = df.sort_values(by=classes[0])
    fig,axes=plt.subplots(1,n_classes, sharey = True)
    if n_classes==1:
        axes=[axes]
    fig.set_size_inches(3*n_classes+ 1, 8)
    #     fig.suptitle("Coefficient of the features")
    fontsize = "x-large" #if n_classes !=1 else "large"
    for i in range(n_classes):
        # Access the row containing the coefficients for this class
        class_coef = df.iloc[:,i]
    #         sort_idx = np.argsort(class_coef)
        colors = [COLORS[7] if c < 0 else COLORS[3] for c in class_coef]
        yticks = np.arange(len(class_coef))
        axes[i].barh(yticks, class_coef, color=colors)#
    #         feature_names = np.array(feature_names)
        # Here I corrected the start to 0 (Your code has 1, which shifted the labels)
        axes[i].tick_params(axis = 'both', labelsize = fontsize) #which = 'minor',
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels(list(df.index)) # rotation=60,fontsize=fontsize  ha="right"
        axes[i].set_title(classes[i],fontsize='xx-large')
        axes[i].set_ylim(-0.6, len(class_coef)-0.4) #bottom: float, top: float
    fig.text(0.5, 0.04, 'Coefficient of the features', ha='center',fontsize='xx-large')
    #'medium', 'large'  'x-large', 'xx-large'
    fig.savefig(filename + ".coef.svg")    

# np.logspace(2, 6, 6, base=2)
def plot_bin_roc_curve(y_test, y_score,class_names,filename):
    """Score(pred)表示每个测试样本属于正样本的概率,从高到低，依次将“Score”值作为阈值threshold，
    当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。
    每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。
    当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。
    将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。
    多分类的可以每个标签绘制一条ROC曲线
    一图一表原则,样品mf命名以数字开头0_no, 1_CD,这样自动0是阴性样品，1是阳性样本
    """
    fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr,tpr)
    df = pd.DataFrame.from_dict(
        dict(zip(['classes', 'fpr', 'tpr', 'auc'],
                 (class_names[1], fpr, tpr, roc_auc))))
    df.to_csv(filename + ".roc_curve.csv",  index=False)
    
    fig = plt.figure(figsize=(8,8))
    lw=2
    ax = fig.add_subplot(111)
    plt.grid(which='major', axis='both')
    ax.plot(fpr, tpr, color='darkorange',
            lw=2, label='{0} (area ={1:0.2f})'.format(class_names[1],roc_auc)) ###假正率为横坐标，真正率为纵坐标做曲线
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right",fancybox=True, framealpha=0.8, fontsize=6)
    fig.savefig(filename + ".roc_curve.svg")
    
     
#     fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1]) 二分类
#roc_auc: GridSearch(est, param_grid, scoring='roc_auc')
# auc_score, by setting the new scoring parameter to roc_auc: GridSearch(est, param_grid, scoring='roc_auc'). It will do the right thing and use predict_proba (or decision_function if predict_proba is not available).


def plot_multi_roc_curve(y_test, y_score,classes,filename):
    """ Learn to predict each class against the other
    Compute ROC curve and ROC area for each class
    classes order same as y_test
    """   
    from scipy import interp
    # 计算每一类的ROC
    n_classes=len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    dfs = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        df = pd.DataFrame.from_dict(
            dict(zip(['classes', 'fpr', 'tpr', 'auc'],
                     (classes[i], fpr[i], tpr[i], roc_auc[i]))))
        dfs.append(df)
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    df = pd.DataFrame.from_dict(
        dict(zip(['classes', 'fpr', 'tpr', 'auc'],
                 ('micro', fpr["micro"], tpr["micro"], roc_auc["micro"]))))
    dfs.append(df)
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
    
    df = pd.DataFrame.from_dict(
        dict(zip(['classes', 'fpr', 'tpr', 'auc'],
                 ('macro', fpr["macro"], tpr["macro"], roc_auc["macro"]))))
    dfs.append(df)
    concat_dfs = pd.concat(dfs)
    concat_dfs.to_csv(filename + ".roc_curve.csv", index=False)
    
    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro_average (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=lw)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro_average (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=lw)
    colors = COLORS[:n_classes]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=' {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=8)
    plt.savefig(filename + ".roc_curve.svg")

######################## pipeline functions  END ###################

    
class SklearnPipeline():
    def __init__(self, X_filename, X, Y, log="SklearnPipeline.log", outdir="./"):
        """load the feature matrix(X) and maping file(Y)
        X should be normalized or relative transformed
        """
        self.outdir = os.path.abspath(outdir)
        self.filename = os.path.join(self.outdir, X_filename)
        utils.create_dir(self.outdir)
        self.logger = utils.create_logger(log)
        self.logger.info("Start reading data from {}".format(X_filename))
        self.X, self.Y  = X, Y
        self.Y = self.Y.loc[self.X.index,:] #sort and select samples
        self.target_names=[re.sub(r" +", "_",name) for name in np.unique(self.Y.values)] #sorted in same order of 0 1 2 labels 
        self.label_counts = Counter(self.Y.iloc[:,0])
        self.logger.info('Finish loading data from {}, dimension is {}, \n \t\t\t label counts {}'.format(
                                            X_filename, self.X.shape, self.label_counts))
        self.stats = []
        self.stats.append(("original_dim", self.X.shape))
    
    @utils.tryExcept
    def filter_low_prevalence_features(self, prevalence=0.2, to_relative=False):
        """pd.DataFrame: rows = feaures
        feature组内覆盖度最大的要大于prevalence, 保证特征在该覆盖度下可以代表该组，防止过滤掉组内特性富集的features
        可以每个组内过滤低覆盖度的，group-specific filter, for update later
        OTU counts were converted to relative abundances, filtered at a minimum of 10% prevalence across samples
        稀疏矩阵如何过滤。
        """
        self.X = self.X.loc[self.X.sum(axis=1)>0 , self.X.sum(axis=0)>0] #del 0 featrues and 0 samples
        if prevalence !=0:
            within_class_prevalence =[np.sum(self.X[self.Y.iloc[:,0].values==k]>0, axis=0)/v for k,v in self.label_counts.items()] 
            # features within_class prevalence for each label list of n_class Series
            if to_relative :
                self.X = (self.X.T / self.X.sum(axis=1)).T # transform to relative abundance matrix, rows are samples        
            self.X = self.X.loc[:, pd.DataFrame(within_class_prevalence).max() > prevalence] #filter low within class prevalence features
            self.X = self.X.loc[self.X.sum(axis=1)>0 ,:] # filter 0 samples 间接删除离群样本
            self.X.to_csv(self.filename + ".filter_{}_prevalence.csv".format(prevalence))
        self.Y = self.Y.loc[self.X.index,:] #sort and select samples after feature selection
        self.Y = LabelEncoder().fit_transform(self.Y.values.ravel())
        self.logger.info("Filtered the features with max within_class prevalence lower than {}, dimension is {}".format(prevalence, self.X.shape))
        self.stats.append(("prevalence_{}_dim".format(prevalence), self.X.shape))
        
    @utils.tryExcept
    def mrmr_feature_select(self, n_selected_features=50):
        """
        Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." JMLR 2012
        select features index[0] is the most important feature
        j_cmi:  basic scoring criteria for linear combination of shannon information term
        j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))  conditional mutual information mrmr gama=0
        
        互信息(Mutual Information)是度量两个事件集合之间的相关性(mutual dependence)。互信息是点间互信息（PMI）的期望值

        MIfy: mutual information between selected features and response y
        """
#         plot_tsne(self.X,Y=self.Y,targets=self.target_names, filename=self.filename +'.before_mrmr_feature_selection')
        n_samples, n_features = self.X.shape
        x=np.array(self.X)
        if n_selected_features and n_features  > n_selected_features:
            # filter half more features or select 50 features int(n_features*percent)  #
#             self.logger.info("selecting {} features using mrmr".format(num_fea))
            idx, j_cmi, MIfy = MRMR.mrmr(x, self.Y, n_selected_features=n_selected_features)  
        else:
            idx, j_cmi, MIfy = MRMR.mrmr(x, self.Y)  #select automatically  may still  remain many features or 
        num_fea = len(idx)
        # obtain the dataset on the selected features
        self.features = self.X.columns[idx].values
        mrmr_report = pd.DataFrame({"features":self.features, "j_cmi":j_cmi, "MIfy": MIfy}, columns=['features', 'j_cmi', 'MIfy'])
        mrmr_report = mrmr_report.sort_values('MIfy',ascending=False)
        mrmr_report.to_csv(self.filename + ".mrmr_features.report.csv",index=False)
        
        self.X = self.X.iloc[:,idx] #select mrmr features
        sel_bools = self.X.sum(axis=1)!=0 # filter  all 0 rows samples.
        self.X = self.X[sel_bools] 
        self.Y = self.Y[sel_bools] 
        self.X.to_csv(self.filename + ".mrmr_sel_features.csv")
        self.logger.info("Selected {} features using mrmr".format(num_fea))
        self.stats.append(("mrmr_dim", self.X.shape))
#         plot_tsne(self.X,Y=self.Y,targets=self.target_names, filename=self.filename +'.after_mrmr_feature_selection')

    
    @utils.tryExcept
    def over_sampling(self):
        """Over-sampling the minority class for imbalance data using SMOTE
        https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/
        The main objective of balancing classes is to either 
        increasing the frequency of the minority class or 
        decreasing the frequency of the majority class. 

        Over-Sampling increases the number of instances in the minority class by 
        randomly replicating them in order to present a higher representation of the minority class in the sample
        
        Disadvantages
        It increases the likelihood of overfitting since it replicates the minority class events.
        In most cases, synthetic techniques like SMOTE and MSMOTE will outperform the conventional oversampling and undersampling methods
        For better results, one can use synthetic sampling methods like SMOTE and MSMOTE 
        along with advanced boosting methods like Gradient boosting and XG Boost.
        
        X G Boost is generally a more advanced form of Boosting and takes care of imbalanced data set by balancing it in itself
        try XG boosting on the imbalanced data directly set to get better results.
        """
        class_sample_count = Counter(self.Y)
        self.stats.append(("ori_class_sample_count", class_sample_count))
        isbalanced = len(set(class_sample_count.values()))
        if isbalanced ==1:
            self.logger.info('The dataset is balanced with class_sample_count {}'.format(class_sample_count))
            self.features = self.X.columns
        else:
            self.logger.info('Dataset shape {} before over sampling'.format(class_sample_count))
            sm = SMOTE(random_state=RANDOM_STATE)
            self.features = self.X.columns
            self.X, self.Y = sm.fit_resample(self.X, self.Y)
            self.X = pd.DataFrame(self.X,columns=self.features)
            self.stats.append(("smote_class_sample_count", Counter(self.Y)))
            self.logger.info('Over sampled dataset with SMOTE, shape {}'.format( Counter(self.Y) ))
            
            
    @utils.tryExcept
    def select_best_scl_clf(self, SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS, 
                            scoring= 'accuracy', outer_cv=10, inner_cv=5, n_jobs=1, search=0):
        """选最好的标准化方法和classifier (default parameters) 组合
        Each sample i.e. each row of the data matrix
        X = X_full[:, [0,1]] #根据前两个特征判断那种数据转化效果好, 用pca后可视化进行比较
        数据转化过滤中同时考虑binarizer 和 abundance based的数据哪种转化方式更能提高分类效果，并考虑分类样本均衡的问题。

        Compare the effect of different scalers on data with outliers
        https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
        test one model with multiple scalers
        This example uses different scalers, transformers, and normalizers to bring the data within a pre-defined range 

        In general, learning algorithms benefit from standardization of the data set. 
        If some outliers are present in the set, robust scalers or transformers are more appropriate. 
        The behaviors of the different scalers, transformers, and normalizers on a dataset containing marginal outliers 
        is highlighted in Compare the effect of different scalers on data with outliers.

        A notable exception are decision tree-based estimators that are robust to arbitrary scaling of the data.
        """
        self.search = search
        scls=[SCALER[0] for SCALER in SCALERS]
        clfs = [clf[0].__class__.__name__ for clf in Other_CLASSIFIERS]
        trees = [clf[0].__class__.__name__ for clf in Tree_based_CLASSIFIERS]
        if search:
            self.logger.info('Select the best tree-based classifiers: {} \n \t\t\t and combination of scalers: {} \n \t\t\t and classifiers: {} \n \t\t\t Tune each classifier with GridSearchCV'.format(trees, scls, clfs))
        else:
            self.logger.info('Select the best tree-based classifiers: {} \n \t\t\t and combination of scalers: {} \n \t\t\t and classifiers: \n \t\t\t with default parameters'.format(trees, scls, clfs))
              
       
        # search best combination with scalars and clfs with default params 
        PARAM_GRID = [
            {
                'scl': [NonScaler()],
                'clf':  [clf[0] for clf in Tree_based_CLASSIFIERS]
            },
            {
                'scl': [scl[1] for scl in SCALERS],
                'clf':  [clf[0] for clf in Other_CLASSIFIERS],
            }
        ]

        # search best combination with scalars and hyper-parameters tuned clfs
        PARAM_GRID_SEARCH =[
            dict({ 'scl': [NonScaler()], 
                  'clf': [clf[0]], 
                  **clf[1] }) for clf in Tree_based_CLASSIFIERS 
        ] + [
            dict({ 'scl': [scl[1] for scl in SCALERS], 
                  'clf': [clf[0]], 
                 **clf[1] }) for clf in Other_CLASSIFIERS
        ]


        pipeline = Pipeline(memory=mem, # memory: store the transformers of the pipeline
                      steps= [
                                ('scl', NonScaler()),
                                ('clf', GaussianNB())
                      ])    
        if search:
            param_grid = PARAM_GRID_SEARCH  # {'scaler': [SCALER[1] for SCALER in SCALERS]} #grid 自动化并行比较多个scalers
        else:
            param_grid = PARAM_GRID
            
        grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                                   scoring = scoring, iid=False, cv =inner_cv,  
                                   n_jobs=n_jobs, verbose=1, return_train_score=True)

        grid_search.fit(self.X, self.Y)# Train the classifier with balanced data ; fit did nothing 
        self.grid_search = grid_search
        self.scoring = scoring
        #通过交叉验证获得预测值
        self.y_pred = cross_val_predict(grid_search, self.X, self.Y, cv=outer_cv, n_jobs=1) #n_job= -1 会出错，和inver_cv冲突
        #for each element in the input, the prediction that was obtained for that element when it was in the test set.
        #In each iteration, label of i'th part of data gets predicted. In the end cross_val_predict merges all partially predicted labels and returns them as the final result.
        self.accuracy_score = '%0.2f' % metrics.accuracy_score(self.Y, self.y_pred)  #balanced_accuracy_score(y_true, y_pred)
        
        self.best_estimator_ = grid_search.best_estimator_  #best  estimator based on Mean accuracy of self.predict(X) wrt. y
        self.best_clf =  self.best_estimator_.named_steps["clf"]
        self.scl_name = self.best_estimator_.named_steps["scl"].__class__.__name__
        self.clf_name = self.best_clf.__class__.__name__
        
        if not search:
            report_topN_cv_results(grid_search.cv_results_,n_top=10,filename=self.filename)
        
        #save cv_results
        df=pd.DataFrame(grid_search.cv_results_)
        df.to_csv(self.filename + ".all.cv_results.csv", index=False)


    @utils.tryExcept
    def hypertune_best_classifier(self, All_CLASSIFIERS, pltcfm=True, outer_cv=10, inner_cv=5,n_jobs=1):
        """compare classifiers by nested Cross-Validation
        hypertune best classifier
        RandomizedSearchCV 来优化胜出的分类器
        n_components == min(n_samples, n_features)[defult]
        n_components=0.85
        variance that needs to be explained is greater than the percentage
        There are more sophisticated ways to choose a number of components,
        of course, but a rule of thumb of 70% - 90% is a reasonable start.
        """
        if self.search:
            'no need to tune again'
            best_hypertuned_clf = self.best_clf
            grid_search = self.grid_search
            
        else:  
            self.logger.info('Hypertune the best classifier {} with GridSearchCV'.format(self.clf_name))
       
            # cross prediction do not need to split the data        
            #X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, random_state=RANDOM_STATE)

            best_clf_index = [i[0] for i in All_CLASSIFIERS].index(self.best_clf)
            param_grid = All_CLASSIFIERS[best_clf_index][1]

            grid_search = GridSearchCV(self.best_estimator_, param_grid=param_grid, 
                               scoring= self.scoring, iid=False, #[independently identically distributed across the folds] return the average score across folds
                               cv=inner_cv, #inner_cv  train/validation dataset default 3
                               n_jobs=n_jobs,verbose=1,return_train_score=True)  #Mean accuracy of self.predict(X) wrt. y

            grid_search.fit(self.X, self.Y) # Train the classifier with balanced data
            self.y_pred = cross_val_predict(grid_search, self.X, self.Y, cv=outer_cv) #outer_cv


            self.accuracy_score = '%0.2f' % metrics.accuracy_score(self.Y, self.y_pred) #balanced_accuracy_score(y_true, y_pred)
            self.best_estimator_ = grid_search.best_estimator_ 
            
            best_hypertuned_clf = grid_search.best_estimator_.named_steps["clf"]

            #save cv_results   
            df=pd.DataFrame(grid_search.cv_results_)
            df.to_csv(self.filename + ".{}.hypertuned.cv_results.csv".format(self.clf_name), index=False)
            
            
        self.logger.info("Best optimized classifier: {} , Accuracy:{}, Best Param:{}".format(
                          self.clf_name, self.accuracy_score, grid_search.best_params_))
       
        self.stats.append(("best_estimator", {k:v.__class__.__name__ for k,v in  grid_search.best_estimator_.named_steps.items()}))
        self.stats.append(('hypertuned_best_parameters', grid_search.best_params_))
        self.stats.append(('hypertuned_best_score_{}'.format(self.scoring), '%0.2f' % grid_search.best_score_))  #mean_test_score
        self.stats.append(('hypertuned_accuracy', self.accuracy_score)) #refit all samples score


        #plot  hypertuned classification report
        report = metrics.classification_report(self.Y, self.y_pred, target_names=self.target_names, output_dict=True)
        filename = self.filename + ".{}.hypertuned".format(self.clf_name)
        plot_classification_report(report, filename=filename)

            
         #save model
        modelf = self.filename + ".{}_{}.model.z".format(self.scl_name, self.clf_name)
        dump(self.best_estimator_, modelf)
#         clf = load(modelf) 
        if pltcfm:
            """plot cunfusion matrix""" 
            cfmf=self.filename + '.{}.hypertuned.confusion_matrix'.format(self.clf_name)
            cfm_html = self.filename + '.{}.hypertuned.PyCM_report'.format(self.clf_name)
            dic=dict(zip(np.unique(self.Y),self.target_names))
            actual_vector = [dic[i] for i in self.Y]
            predict_vector = [dic[i] for i in self.y_pred]
            cm = ConfusionMatrix(actual_vector=actual_vector, predict_vector=predict_vector)   # pycm
            cm.save_html(cfm_html) # cross prediction result
            cfm = metrics.confusion_matrix(self.Y, self.y_pred)
            cfm_df=pd.DataFrame(cfm, columns=self.target_names,index=self.target_names)
            plot_confusion_matrix(cfm_df, filename=cfmf, accuracy = self.accuracy_score)
            
        #roc_auc_score = metrics.roc_auc_score(y_test, self.y_pred)  #roc 不支持多分类
        
        # plot roc courve 
        # Yet this only counts for SVC where the distance to the decision plane is used to compute the probability - therefore no difference in the ROC.
        # refit all overfit
        y_proba = None
        if hasattr(best_hypertuned_clf, "decision_function"):
            y_proba = grid_search.decision_function(self.X) 
            # decision_function, finds the distance to the separating hyperplane.
            # y_proba = cross_val_predict(grid_search, self.X, self.Y, cv=outer_cv, method='decision_function')
        elif hasattr(best_hypertuned_clf, "predict_proba"):
            # predict_proba is a method of a (soft) classifier outputting the probability of the instance being in each of the classes.
            # y_proba = cross_val_predict(grid_search, self.X, self.Y, cv=outer_cv, method='predict_proba')
            y_proba = grid_search.predict_proba(self.X)[:, 1]
        
        if y_proba is not None:   # elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
            if len(self.target_names)==2:
                plot_bin_roc_curve(self.Y, y_proba, self.target_names, filename)
            else:
                y_test = LabelBinarizer().fit_transform(self.Y) 
                plot_multi_roc_curve(y_test, y_proba, self.target_names, filename)
        
   
        #plot topK important features and tsne scatter
        n_features = self.X.shape[1]
        k = 20 if n_features > 20 else n_features
        
        if hasattr(best_hypertuned_clf, "feature_importances_"):
            """plot feature importance"""
            feature_importances = best_hypertuned_clf.feature_importances_
           
            csv = filename + ".feature_importance.csv"
            df = pd.DataFrame({
                'feature_names': self.features,
                'importance': feature_importances,
            })
            df.sort_values(by='importance',ascending=False).to_csv(csv, index=False)
            
            svg = filename + ".top{}.feature_importance.svg".format(k)
            top_k_feature_names = plot_feature_importance(df, k, svg)
#             topk_X = self.X[top_k_feature_names]
#             topk_X = topk_X.loc[topk_X.sum(axis=1)>0, : ] #del 0 samples and 0 featrues topk_X.sum(axis=0)>0
#             topK_Y = self.Y[[i for i,j in enumerate(self.X.index) if j in topk_X.index]] #sort and select samples sel_bools = self.X.sum(axis=1)!=0 #
#             plot_tsne(topk_X,Y=topK_Y,targets=self.target_names, filename=self.filename +'.{}.hypertuned.top{}.important_features'.format(self.clf_name, k))

        elif hasattr(best_hypertuned_clf, "coef_"):
            coef =best_hypertuned_clf.coef_      # As the second of the categories is the Yes category
            #https://www.displayr.com/how-to-interpret-logistic-regression-coefficients/
            if coef.shape[0]==1:
                df=pd.DataFrame(coef.reshape(1,-1),index=[self.target_names[1]], columns=self.features)
            else:
                df=pd.DataFrame(coef,index=self.target_names,columns=self.features)
            df.T.to_csv(filename + ".coef.csv") 
            plot_coefficients(df.T, topk=20, filename=filename)
      
        stats_df = pd.DataFrame({"stats_index":[i[0] for i in self.stats], "stats_value":[str(i[1]) for i in self.stats]})
        stats_df.to_csv(self.filename + ".log_stats.csv",index=False)
        self.logger.info("Pipeline is finished")
        
           
    def make_predictions(self, test_df, model):
        '''test_df is in DataFrame format with index being samples'''
        load_model = load(model)       
        predicts = load_model.predict(X)
        predict_labels = [self.target_names[i] for i in predicts]
        result = pd.DataFrame({'SampleID': test_df.index, 'predicted label': predict_labels})
        return result 


def main(X_filename, X, Y, SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS, cv=5, search=0, 
         log="SklearnPipeline.log", outdir="./"):
    fe = SklearnPipeline(X_filename, X, Y, log=log, outdir=outdir) 
    fe.filter_low_prevalence_features(percent=0.1)
    fe.mrmr_feature_select(percent=0.5)
    fe.over_sampling()
    fe.select_best_scl_clf(SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS,
                           outer_cv=10, inner_cv=5, n_jobs=-1, search=search_tune)
    tune = 0 if search else 1 # if each classifier was tuned, no need to tune the best clf 
    fe.hypertune_best_classifier(search=tune, pltcfm=True, inner_cv=cv)
    
  
    
if __name__ == '__main__':
    
    ###################### argparse ###########################
    import argparse
    parser = argparse.ArgumentParser(
        description="A pipeline for automatically identify the best performing combinations of scalars and classifiers for microbiomic data",
        usage="%(prog)s --help or -h",
        epilog="""
        Example:
           python  %(prog)s Gevers2014_IBD_ileum.csv Gevers2014_IBD_ileum.mf.csv --mrmr_n 20 --over_sampling  --outdir ./ 
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
       
    parser.add_argument('X_file', help='feature matrix file (required)') 
    parser.add_argument('Y_file', help='map file (required)')
    parser.add_argument('--outdir', '-o', default='./', help='path to store analysis results, default=\'./\'')
    
    parser.add_argument('--prevalence', '-p', default=0.2, type=float, help='filter low within-class prevalence features, default= 0.2')
    parser.add_argument('--mrmr_n', default=0, type=int, help='number of features selected with MRMR, default=0')
    parser.add_argument('--over_sampling', action='store_true', help='over-sampling with SMOTE') 
    
    parser.add_argument('--search', action='store_true', help='tune parameters of each classifier while selecting the best scaler and classifier') 
    parser.add_argument('--outer_cv', default=10, type=int, help='number of fold in the outer loop of nested cross validation default=10')
    parser.add_argument('--inner_cv', default=5, type=int, help='number of fold in the inner loop of nested cross validation, default=5')
    parser.add_argument('--scoring', default='accuracy', help='one of ccuracy, average_precision, f1, f1_micro, f1_macro, f1_weighted, f1_samples, neg_log_loss, precision, recall, roc_auc, default=accuracy')
    parser.add_argument('--n_jobs', '-j', default=1, type=int, help='number of jobs to run in parallel, default= 1')
    args=parser.parse_args()
    ##########################################################
    
    now = utils.now()
    X_filename = os.path.basename(args.X_file)
    log = "{}_{}.log".format(X_filename, now)
    sec = time()

    X, Y =  get_data(args.X_file, args.Y_file)
    skp = SklearnPipeline(X_filename, X, Y, log=log, outdir=args.outdir) 
    skp.filter_low_prevalence_features(prevalence=args.prevalence)
    
    if args.mrmr_n:
        skp.mrmr_feature_select(n_selected_features=args.mrmr_n)
    if args.over_sampling:        
        skp.over_sampling()
        
    skp.select_best_scl_clf(SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS, 
                            outer_cv=args.outer_cv, inner_cv=args.inner_cv, n_jobs=args.n_jobs, search=args.search) #default parameters select best combination
    skp.hypertune_best_classifier(All_CLASSIFIERS, pltcfm=True, outer_cv=args.outer_cv, inner_cv=args.inner_cv, n_jobs=args.n_jobs)
    skp.logger.info('sklearn pipeline finished, total time cost: {} s'.format(round(time()-sec, 1)))
    
#   Python 3.3 and will be removed from Python 3.8: use time.perf_counter or time.process_time instead 
