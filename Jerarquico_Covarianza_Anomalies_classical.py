#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import math
import numpy as np
import skfeature
from skfeature.utility.mutual_information import su_calculation

# #### Funciones Auxiliares
#Funcion para vectorizar
def vectorizar(df, label):
    inCols = df.columns
    if label!='':
        #inCols.pop(label)
        print(inCols)
        for c in label:
            inCols.remove(c)
        print(inCols)
    assembler = VectorAssembler(inputCols=inCols, outputCol='features')
    df_vec = assembler.transform(df)   
    df_vec.select('features').show()
    return df_vec


def fill_with_mean(this_df, include=set()):
    stats = this_df.agg(*(avg(c).alias(c) for c in this_df.columns if c in include))
    #print (stats)
    return this_df.na.fill(stats.first().asDict())

def fill_with_xtrem(this_df, include=set()):
    tdf1= this_df[include]
    tdf2= this_df.drop(include,axis=1)
    tdf1=tdf1.fillna([-1]*tdf1.max())
    this_df=pd.concat([tdf2,tdf1], axis=1)   
    return this_df

def fcbf(X, y, **kwargs):
    """
    This function implements Fast Correlation Based Filter algorithm
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        delta: {float}
            delta is a threshold parameter, the default value of delta is 0
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    SU: {numpy array}, shape (n_features,)
        symmetrical uncertainty of selected features
    Reference
    ---------
        Yu, Lei and Liu, Huan. "Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution." ICML 2003.
    """

    n_samples, n_features = X.shape
    if 'delta' in kwargs.keys():
        delta = kwargs['delta']
    else:
        # the default value of delta is 0
        delta = 0

    # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
    t1 = np.zeros((n_features, 2))
    #t1 = np.zeros((n_features, 2), dtypes='object')
    for i in range(n_features):
        f = X[:, i]
        t1[i, 0] = i
        t1[i, 1] = su_calculation(f, y)
    print("T1",t1)
    s_list = t1[t1[:, 1] > delta, :]
    print("SLIST",s_list)
    # index of selected features, initialized to be empty
    F = []
    # Symmetrical uncertainty of selected features
    SU = []
    while len(s_list) != 0:
        # select the largest su inside s_list
        idx = np.argmax(s_list[:, 1])
        print("IDX: ",idx)
        # record the index of the feature with the largest su
        print("X\n",X)
        fp = X[:, int(s_list[idx, 0])]
        prin("FP",fp)
        np.delete(s_list, idx, 0)
        F.append(s_list[idx, 0])
        SU.append(s_list[idx, 1])
        for i in s_list[:, 0]:
            i=int(i)
            fi = X[:, i]
            if su_calculation(fp, fi) >= t1[i, 1]:
                # construct the mask for feature whose su is larger than su(fp,y)
                idx = s_list[:, 0] != i
                idx = np.array([idx, idx])
                idx = np.transpose(idx)
                # delete the feature by using the mask
                s_list = s_list[idx]
                length = len(s_list)//2
                s_list = s_list.reshape((length, 2))
    return np.array(F, dtype=int), np.array(SU)



def su(X, y, **kwargs):
    """
    This function implements Fast Correlation Based Filter algorithm
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        delta: {float}
            delta is a threshold parameter, the default value of delta is 0
    Output
    ------
    s-list: {numpy array}, shape (n_features,)
        symmetrical uncertainty of selected features
    """
    n_samples, n_features = X.shape
    if 'delta' in kwargs.keys():
        delta = kwargs['delta']
    else:
        # the default value of delta is 0
        delta = 0

    # t1[:,0] stores index of features, t1[:,1] stores symmetrical uncertainty of features
    t1 = np.zeros((n_features, 2))
    #t1 = np.zeros((n_features, 2), dtypes='object')
    for i in range(n_features):
        f = X[:, i]
        t1[i, 0] = i
        t1[i, 1] = su_calculation(f, y)
    print("T1",t1)
    s_list = t1[t1[:, 1] > delta, :]
    return s_list



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sPCA
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
import os
import pickle

class covTaxonomy:
    
    Pesos=None
        
    def __init__(self, df, indexCol, disclus, disEval):
        self.dfk_means = df 
        self.dfk_means.index=df[indexCol]
        self.dfk_means=self.dfk_means.drop([indexCol],axis=1)
        self.disClus = disclus
        self.disEval = disEval
        self.indexCol = indexCol
        self.maxK = 15        
        self.modelosKMeans={}

    #Vectorizar y escalar
    def _vectorize_scale(self,df_v,noVec):
        dataCols = list(df_v.columns)
        try:
            dataCols.remove(self.indexCol)
        except:
            a=1
        for c in noVec:
            dataCols.remove(c)
        noVecCols=[]
        '''for c in df_v.columns:
            if c not in dataCols:
                noVecCols=noVecCols + [c]
        '''
        print("no vectorizar:", noVecCols)
        df_Vec=df_v[dataCols]
        df_noVec=df_v.drop(dataCols,axis=1)
        #print(dataCols)
        print("columnas",len(df_v.columns))
        print("Vectorizar DataCols", len(dataCols))
        scaler = StandardScaler()
        scaler.fit(df_Vec)
        #print("ANTES DE VECTORIZAR",df_k[dataCols[:1]].describe())
        #print(scaler.mean_)
        #print(scaler.transform(df_Vec).shape())
        df_ks=pd.DataFrame(scaler.transform(df_Vec),columns=[dataCols])
        '''
        #df_ks.index=df_k.index        
        print(df_ks.shape())
        #print("DESPUES DE VECTORIZAR",df_ks[dataCols[:1]].describe())
        '''
        print(df_noVec.columns)
        if len(df_noVec.columns)>0:
            df_ks=pd.concat([df_noVec,df_ks],axis=1)
            columnasfinales=list(df_noVec.columns) + dataCols
            df_ks.columns = df_ks.columns.get_level_values(0)
            df_ks.columns=columnasfinales
        
        print(df_ks.head())
        df_ks.columns = df_ks.columns.get_level_values(0)
        
        #print(dldldlld)

        return df_ks, scaler

    def kMeansBest(self, df):
        print("ENCONTRAR MEJOR k")
        #df.show()
        print("NULLOS", df.isnull().any().any())
        print("NULOS",df.isnull().sum())
        #df.show()
        minDist=0
        kMin = 0
        codo = 0
        fin = 0
        df=df.copy()
        for i in range(2,self.maxK):
            print(f"N_CLUSTERS={i}")
            kmeans = KMeans(n_clusters=i,random_state=42)
            Kmeans_model = kmeans.fit(df)
            # transform your initial dataframe to include cluster assignments
            TestDataProfile = Kmeans_model.transform(df)
            # Evaluate clustering by computing Silhouette score
            print("CALCULANDO SILHOUETTE")
            silhouette = silhouette_score(df, kmeans.labels_)
            print("Silhouette for k="+str(i)+" distances " + str(silhouette))
            if (silhouette > minDist):
                minDist = silhouette
                kMin = i
                print("k para max silhouette ="+str(i))
                codo=0
            else:
                if (silhouette <= minDist):
                    codo=codo+1
                    if codo>=3:
                        break
        #self.kMin = kMin
        return kMin,minDist
    
    
    #Aplicar el k-Means
    def _kMeans_apply(self, kMin, df):
        try:
            df=df.drop([self.indexCol],axis=1)
        except:
            a=1
        kmeans = KMeans(n_clusters=kMin,random_state=42)
        Kmeans_model = kmeans.fit(df)
        # transform your initial dataframe to include cluster assignments
        TestDataProfile = pd.DataFrame(Kmeans_model.labels_,columns=['prediction'])
        #TestDataProfile.select('prediction').show()
        #print(TestDataProfile)
        print(TestDataProfile.groupby(['prediction']).size())
        TestDataProfile.index=df.index
        TestDataProfile=pd.concat([df,TestDataProfile],axis=1)
        print("columnas etiquetadas", len(TestDataProfile.columns))
        return Kmeans_model, TestDataProfile
    
    #CALCULAR LA IMPORTANCIA DE LAS VARIABLES POR ENTROPIA
       
    def _feature_importances_su(self,df):
        classAt = df.columns.get_loc("prediction")
        varCols=list(df.columns)
        varCols.remove("prediction")
        X_train=df.drop(['prediction'],axis=1)
        X_trainColumns=list(X_train.columns)
        y_train=df['prediction']
        #X_train, y_train=fcbf_wrapper(df,classAt)
        X_train=X_train.to_numpy()
        y_train=y_train.to_numpy()
        thr=len(X_trainColumns)/2
        #entropias=fcbf(X_train,y_train, thr)
        entropias=su(X_train,y_train)
        print("Entropias",entropias)
        print("Tipo",type(entropias))
        entropy=pd.DataFrame(entropias)
        entropy.columns=['colVariable','valor']
        entropy['variable']=varCols
        print(entropy)
        entropy['importance_norm']=entropy['valor'].apply(lambda x: x/(entropy['valor'].max()))
        entropy=entropy.drop(['colVariable'],axis=1)
        print("ENTROPIA", entropy)        
        return entropy
        
    def _feature_importances_fcbf(self, df):
        #print("ANTES DE FCBF",df.columns)
        classAt = df.columns.get_loc("prediction")
        X_train=df.drop(['prediction'],axis=1)
        X_trainColumns=list(X_train.columns)
        y_train=df['prediction']
        #X_train, y_train=fcbf_wrapper(df,classAt)
        X_train=X_train.to_numpy()
        y_train=y_train.to_numpy()
        thr=len(X_trainColumns)/2
        #entropias=fcbf(X_train,y_train, thr)
        entropias=fcbf(X_train,y_train)
        print("Entropias",entropias,"\n",entropias[0],"\n",entropias[1])
        entropy=pd.DataFrame()
        for v0,v1 in zip(entropias[0],entropias[1]):
            print(v0,v1)
            e=pd.DataFrame([[v0,v1]],columns=['colVariable','valor'])
            entropy=entropy.append(e)
        #entropy=pd.DataFrame(entropias[0],entropias[1],columns=['valor','colVariable'])
        entropy['variable']=entropy['colVariable'].apply(lambda x: list(X_trainColumns)[int(x)])
        entropy['importance_norm']=entropy['valor'].apply(lambda x: x/(entropy['valor'].max()))
        entropy=entropy.drop(['colVariable'],axis=1)
        #print("ENTROPIA", entropy)        
        return entropy
        

    
    #CALCULAR LA IMPORTANCIA DE LAS VARIABLES CON UN GBTCLASSIFIER
    def _feature_importances(self, df):
        clf = GradientBoostingClassifier(random_state=0)
        X_train=df.drop(['prediction'],axis=1)
        y_train=df['prediction']
        clf.fit(X_train, y_train)
        '''
        df_rf = df.withColumnRenamed('prediction','label')
        print(df_rf.columns)
        stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = stringIndexer.fit(df_rf)
        td = si_model.transform(df_rf)
        print("columnas indexadas", len(td.columns))
        rf = RandomForestClassifier(numTrees=10, maxDepth=10,labelCol="indexed", seed=42)
        model_rf = rf.fit(td)
        print("Columnas post RF", len(td.columns))
        '''
        
        fi = pd.DataFrame([clf.feature_importances_])
        fi=fi.transpose()
        fi.columns=['valor']
        #campos = pd.DataFrame(df_rf.columns[:(len(df_rf.columns))])
        campos = pd.DataFrame(X_train.columns)
        fi=pd.concat([fi,campos],axis=1)
        fi.columns = ['valor','variable']
        print("Campos",fi)
        fi = fi.dropna()
        fi_red = fi[fi['valor']>0.0001]
        print(fi_red.sort_values(by='valor',ascending='True'))
        df_fi_max = fi_red['valor'].max()
        print("max",df_fi_max)
        fi_red['importance_norm'] = fi_red['valor'].apply(lambda x: x/df_fi_max)
        df_fi = fi_red.sort_values(by='importance_norm',ascending=False)
        if len(fi_red)>0:
            #df_fi = spark.createDataFrame(fi_red)
            #df_fi = df_fi.withColumn('Level', lit(bucle))
            #df_fi = df_fi.withColumn('K', lit(kMin))
            #df_fi = df_fi.withColumn('EvalK', lit(minDist))
            #df.agg(F.max(F.abs(df.v1))).first()[0]        print(df_fi_max)
            return df_fi

    def _getDummys(self,df,v):
        categ = df[v].to_list()
        exprs = [when(col(v) == cat,1).otherwise(0)                 .alias(v+"_"+str(cat)) for cat in categ]
        df = df.select(exprs+df.columns)
        df.select(exprs).show(5)
        return df
        
    def _agglomerative_clustering(self):
        #pesos = self.Pesos.select('features').toPandas()   
        pesos=self.Pesos
        print(pesos)
        ''' Version PLAGEMODA
        series = pesos['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
        features = np.apply_along_axis(lambda x : x[0], 1, series)
        #print (features)
        '''
        '''
        #VERSION DAVID
        #lenFeatures=len(np.array(pesos['features'].iloc[0].toArray()))
        lenFeatures=len(pesos.columns)

        #series = pesos['features'].apply(lambda x : np.array(x.toArray()))
        #series = pesos['features'].apply(lambda x : np.reshape(np.array(x.toArray()),[1,5]))

        # from pyspark.sql.functions import udf, col
        #from pyspark.sql.types import ArrayType, DoubleType

        series = tc1.Pesos.select('features').withColumn("f", to_array(col("features"))).select(['features'] + [col("f")[i] for i in range(lenFeatures)])

        features = series.drop('features').toPandas()
        '''
        features=pesos
        features.index=pesos['variable']
        features=features.drop(['variable'],axis=1)
        print(features)
        #Hasta aqui version DAVID
        clustering = AgglomerativeClustering(n_clusters=5).fit(features)
        for c in clustering.labels_:
            print(c)
        #pd.DataFrame({'Column1': clustering.labels_[:, 0]})
        clusters=pd.DataFrame(columns=['cluster'],data=clustering.labels_.flatten(),index=features.index) 
        print(clusters)
        #pesos = tc1.Pesos.drop('features').drop('vecFeatures').toPandas()
        pesosClus = pd.concat([pesos, clusters], axis=1)
        return pesosClus

    def _get_Stats_cluster(self, df):
        #Devolvemos los modelos de vectorizacion y scaler en un diccionario
        stats={}
        print(df.columns)
        #time.sleep(60)
        featuresList=[c for c in df.columns if 'features' in c]
        for c in featuresList:
            df=df.drop(c)
        clusters=list(set(list(df['prediction'])))
        for c in clusters:
            print("ESTADISTICAS PARA CLUSTER",c)
            dfc=df[df['prediction']==c]
            dfc_Vec, scal=self._vectorize_scale(dfc,['prediction'])
            stats.update({c:{'scaler':scal}})
        return stats
    
    def _modelar_kMeans_niveles(self):
        df=self.Niveles
        self.dfk_NivLabeled=None
        niveles=sorted(set(list(df['Nivel'])))
        print (niveles)
        for niv in niveles:
            niv=int(niv)
            #Seleccionar columnas del nivel
            '''
            collected = gc.collect()
            print ("Garbage collector: collected %d objects." % collected)
            '''
            sNiv = str(niv).replace('.0','')
            print("NIVEL ",sNiv)
            varNivel = list(df[df['Nivel']==niv]['variable'])
            print(varNivel)
            '''
            lisVar=[self.indexCol]
            for var in varNivel:
                lisVar=lisVar+[var]
            '''
            df_kVar = self.dfk_means[varNivel]
            print(df_kVar.columns)            
            df_kVar_Vec, scal = self._vectorize_scale (df_kVar, '')
            #self.dfk_means.show()
            #Encontrar mejor k para kMeans
            k, dist = self.kMeansBest(df_kVar_Vec)
            #Aplicar el k-Means
            #Ejecutar KMeans
            df_kVar_Vec.describe()
            km_model, dfk_labeled = self._kMeans_apply(k,df_kVar_Vec)
            print("Estadisticas segmentos para nivel", niv)
            stats=self._get_Stats_cluster(dfk_labeled)
            #Guardar Modelo resultado
            self.modelosKMeans.update({niv:{'vars':varNivel,'kmeans_model':km_model,'scaler':scal, 'clusters_stats':stats}})
            #Guardar los datos etiquetados para pasar a la agrupacion de niveles
            print("Antes de guardar los datos etiquetados")
            dfk_labeled=dfk_labeled[['prediction']]
            dfk_labeled['nivel_'+sNiv]=dfk_labeled['prediction']
            dfk_labeled=dfk_labeled.drop(['prediction'],axis=1)
            print(dfk_labeled.columns)
            try:
                self.dfkNivLabeled=self.dfkNivLabeled.join(dfk_labeled,[self.indexCol])
            except:
                self.dfkNivLabeled=dfk_labeled
 
    def jerarquia(self):
        #OBTENCION DE COMPONENTES PRINCIPALES 
        #Vectorizar
        fin=0
        dfk_Original=self.dfk_means
        #print(self.dfk_means.columns)
        disClus, disEval = self.disClus, self.disEval
        self.dfk_means, scal = self._vectorize_scale (self.dfk_means, '')
        #PCA
        #ESTOS PCA NO SIRVEN PARA NADA, NO SE TRABAJA CON ELLOS
        '''
        print("COMIENZA PCA")
        pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
        modelPCA = pca.fit(self.dfk_means)
         
        result = modelPCA.transform(self.dfk_means)
        for p in range(3):
            valPca=udf(lambda v:float(v[p]),FloatType())
            result = result.withColumn('pca_'+str(p),valPca(col('pcaFeatures')))
        
        #result.show()
        pcaCols=[self.indexCol]
        for c in result.columns:
            if "pca" in c:
                pcaCols = pcaCols + [c]
        result = result.select(pcaCols).drop('pcaFeatures')
        #result = model.transform(self.dfk_means).select("pcaFeatures")
        #result.show(5)
        #print(result.columns)
        '''
        print(self.dfk_means.head(3))
        
        
        kmModel, dfk_labeled = self._kMeans_apply(5, self.dfk_means)
        df_fi = self._feature_importances(dfk_labeled)
        print('ORDENADOS', df_fi)
        #df_fi.sort('valor',ascending=False).limit(10).select('variable').show()
        listaVar=list(df_fi[:20]['variable'])        
        #print("LISTAVAR", listaVar)
        #listaVar = listaVar + [indexCol,'prediction']
        listaVar = listaVar + ['prediction']
        dfk_red = dfk_labeled[listaVar]
        dfk_red = pd.get_dummies(dfk_red, columns=['prediction'],prefix='prediction')
        #print(dfk_red)        
        for l in dfk_red.columns:        
            dfk_W = dfk_red
            #print(dfk_W.describe())
            if 'prediction_' in l:
                print("\nCLUSTER ", l,"\n")
                clu=l[11:]
                #print("CLUSTER", clu)
                dfk_W['prediction']=dfk_W[l]
                dfk_W=dfk_W.drop([l],axis=1)
                #print(dfk_W)
                for c in dfk_W.columns:
                    if ('prediction_' in c):
                        dfk_W = dfk_W.drop([c],axis=1)
                #print("DFK_weight", dfk_W.columns)
                dfk_W,a = self._vectorize_scale(dfk_W, ['prediction'])
                #print("INPUT COLS VECTORIZADOR", a.getInputCols())
                #print("Despues de vectorizar",dfk_W)
                #print(dfk_W.describe())
                print(dfk_W.columns)
                #print(dfk_W['prediction'].unique())
                df_fi_W=self._feature_importances_su(dfk_W)
                print ("IMPORTANCIAS su",df_fi)
                
                #print ("\n#Features selected: {0}".format(len(df_fi_W)))
                #print("Selected feature indices:\n{0}".format(df_fi_W))
                #print("importancia por incertidumbre", df_fi_W)
                #df_fi_W=pd.DataFrame(df_fi_W, columns=['importance','colOrder'])
                #print (df_fi_W)
                #maxImportance=df_fi_W['importance'].max()
                #df_fi_W['importance_norm']=df_fi_W['importance'].apply(lambda x: x/maxImportance)
                df_fi_W['peso_'+clu]=df_fi_W['importance_norm']
                df_fi_W=df_fi_W.drop(['importance_norm'],axis=1)
                print(df_fi_W)
                try:
                    #pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,                    left_index=False, right_index=False, sort=True,         suffixes=('_x', '_y'), copy=True, indicator=False,         validate=None)
                    Pesos = pd.merge(Pesos,df_fi_W,on=['variable'],how="outer").drop(['valor'],axis=1)
                except:
                    Pesos = df_fi_W.drop(['valor'],axis=1)
                print(Pesos)
        Pesos.fillna(0)
        #Pesos = Pesos.filter(~col('variable').contains('pca_')).fillna(0)
        Pesos = Pesos.fillna(0)
        print(Pesos)
        #Clustering jerarquico
        Pesos_v,a = self._vectorize_scale(Pesos,['variable'])
        print(Pesos_v)
        self.Pesos=Pesos_v
        #Pesos_v.select('features').show()
        sPesos_pandas = Pesos_v #.select('features').toPandas()
        
        print(sPesos_pandas)
        '''
        for k in range(2,8):
            print("K bisecting", k)
            
            bkm = BisectingKMeans(minDivisibleClusterSize=1).setK(k).setSeed(42)
            model = bkm.fit(Pesos_v.select('features'))
            centers = model.clusterCenters()
            print("CENTROIDES",centers)
            print(model.hasSummary)
            summary = model.summary
            print(summary)
            dfk_pesos = model.transform(Pesos_v)
            dfk_pesos.select('prediction').distinct().show()
        '''    
        #Clustering jerarquico bottom-up aplanado
        
        pesos_Clus=self._agglomerative_clustering()
        print(pesos_Clus)
        
        self.pesos_Clus = pesos_Clus
        #Pivot
        #pivot_pesos_Clus = pesos_Clus.pivot('cluster')
  
        #Varianza que explica cada cluster
        varClus=None
        
        #return
        
        for clus in pesos_Clus.cluster.unique():
            print(c)
            pesos_c = pesos_Clus[pesos_Clus['cluster']==clus]
            #print (pesos_c.columns)
            pesos_ct=pesos_c.drop(columns=['cluster']).transpose()
            #pesos_ct.columns=pesos_ct[0,:]
            print(pesos_ct.columns)

            print(pesos_ct)
            
            cols = pesos_ct.iloc[0].tolist()
            cols
            pesos_ct=pesos_ct.iloc[1:,]
            pesos_ct.columns=cols
            pesos_ct
                   
            #Determinar pesos
            dfIndex=pd.DataFrame(pesos_ct.index)
            dfIndex.index=pesos_ct.index

            new_pesos = pd.concat([dfIndex,pesos_ct],axis=1)

            new_pesos

            new_pesos.iloc[:,1:]
            new_pesos_X=new_pesos.iloc[:,1:].to_numpy()
            new_pesos_Y=new_pesos.iloc[:,0].to_list()

            print(new_pesos_Y)
            print(new_pesos.index)

            clf = DecisionTreeClassifier()

            clf.fit(new_pesos_X, new_pesos.index)
            feat_importance = clf.tree_.compute_feature_importances(normalize=False)
            print("feat importance = " + str(feat_importance))
            
            #Obtener varianza de las columnas determinantes para el cluster
            colsToSee = new_pesos.drop([0], axis=1).columns.tolist()
            print(colsToSee)
            colsToSee = new_pesos.drop([0], axis=1).columns.tolist()
            #print(colsToSee)
            df_k_Pesos=df_k[colsToSee]

            colsToPca=[]
            for c,f in zip(colsToSee,feat_importance):
                print(c,f)
                if f>0:
                    colsToPca=colsToPca + [c]
                    print(colsToPca)
                    
            
            df_k_Pesos=dfk_Original[colsToPca]
            df_k_PesosPCA, scal = self._vectorize_scale (df_k_Pesos, '')
            print(df_k_PesosPCA)
            self.df_k_PesosPCA = df_k_PesosPCA
            print(colsToPca, len(colsToPca))
            #if len(colsToPca)>=3:
            #try:
            '''
            p_pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")
            P_modelPCA = pca.fit(df_k_PesosPCA)
            print("VARIANZAS", P_modelPCA.explainedVariance)
            print("EXITO PCA")
            '''
            '''             
            #vERSION PLAGEMODA
            print("iNTENTO CON SKLEARN")                
            print(df_k_PesosPCA)
            p_df_k_PesosPCA=df_k_PesosPCA.select('features').toPandas()
            series = p_df_k_PesosPCA['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
            print("SERIES", series)
            X=np.apply_along_axis(lambda x : x[0], 1, series)
            print("XXXX", X)
            #Hasta aquí PLAGEMODA
            
            #Version DAVID
            print(pesos)
            pesos = df_k_PesosPCA.select('features').toPandas()
            lenFeatures=len(np.array(pesos['features'].iloc[0].toArray()))

            #series = pesos['features'].apply(lambda x : np.array(x.toArray()))
            #series = pesos['features'].apply(lambda x : np.reshape(np.array(x.toArray()),[1,5]))
            
            series = tc1.Pesos.select('features').withColumn("f", to_array(col("features"))).select(['features'] + [col("f")[i] for i in range(lenFeatures)])
            X = series.drop('features').toPandas()
            #Hasta aquI version DAVID
            '''
            #X = np.array(df_k_PesosPCA.select('features').collect())
            X=df_k_PesosPCA
            pca = sPCA()
            print(X)
            pca.fit(X)
            #sPCA(n_components=2)
            print(pca.explained_variance_ratio_)
            print(pca.singular_values_)
            i=1
            acum=0
            for var in pca.explained_variance_ratio_:
                acum = acum + var
                #print (acum)
                varAcum = pd.DataFrame([['PC'+str(i),acum,clus]])
                i=i+1
                #print (varAcum)
                varClus=pd.concat([varClus,varAcum],axis=0)
                if acum > 0.99999999:
                    break
                #varClus.columns=['Component','Cumulative','cluster']
            print(varClus)
        
        # Determinar niveles
        print(varClus.columns)
        varClus.columns=['Component','Variance','cluster']
        print(varClus)
        #sumVarClus = varClus.groupby('cluster').agg({'Variance': ['sum', 'mean', 'max']})
        sumVarClus = varClus.groupby('cluster').agg({'Variance': ['sum']})
        print(sumVarClus)
        sumVarClus = sumVarClus.reset_index()
        print(sumVarClus)
        sumVarClus.columns = ['_'.join(col).strip() for col in sumVarClus.columns.values]
        print(sumVarClus.columns)
        sumVarClus['Nivel']= sumVarClus['Variance_sum']
        sumVarClus['Nivel'] = sumVarClus['Nivel'].apply(lambda x: 9 if x <=1 else x)
        sumVarClus = sumVarClus.sort_values('Nivel', ascending=False)
        print(sumVarClus)
        idx=0
        for val in sumVarClus['Nivel'].unique():
            print(val)
            sumVarClus['Nivel'] = sumVarClus['Nivel'].apply(lambda x: idx if x==val else x)
            print("Recodificando Niveles\n",sumVarClus)
            idx=idx+1
        print("NIVELES\n",sumVarClus)
        print(varClus.columns)
        self.varClus = varClus
        pesosClus = tc1.pesos_Clus
        pesosClus = pesosClus[['variable','cluster']]
        #Unir con los cluster de variables
        print("PESOS CLUS",pesosClus)
        joinClus = sumVarClus.join(pesosClus.set_index('cluster'),on='cluster_')
        print("NIVELES Y VARIABLES \n", joinClus)
        
        self.Niveles = joinClus[['variable','cluster_','Nivel']]
        self.Niveles.columns=['variable','cluster','Nivel']
        
        #Modelar normalidad niveles de tramas        
        self._modelar_kMeans_niveles()
    
    def _store_model(self,model,fileName):  
        #with open(fileName, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(model, open(fileName, 'wb'))
            
    def store(self,modelsPath):
        try:
            os.chdir(modelsPath)
        except:
            os.mkdir(modelsPath)
            os.chdir(modelsPath)
    
        jerarquia=self.modelosKMeans
        print("Estamos en ", os.getcwd())
        jerarquiaDict={}
        for nivel in jerarquia:
            nivelDict={}    
            #Las variables se guardan directamente
            nivelDict.update({'vars':jerarquia[nivel]['vars']})
            #GuardarModelo y path
            modelName='kmeans_model_'+str(nivel)
            modelName=os.path.join(modelsPath,modelName)
            self._store_model(jerarquia[nivel]['kmeans_model'],modelName)
            nivelDict.update({'kmeans_model':modelName})
            '''
            #Assembler
            modelName=str(jerarquia[nivel]['assembler'].uid)
            modelName=os.path.join(modelsPath,modelName)
            jerarquia[nivel]['assembler'].write().overwrite().save(modelName)
            nivelDict.update({'assembler':modelName})
            '''
            #Scaler
            modelName='scaler_'+str(nivel)
            modelName=os.path.join(modelsPath,modelName)
            self._store_model(jerarquia[nivel]['scaler'],modelName)
            nivelDict.update({'scaler':modelName})
            #Modeler por cluster
            clusterDict={}
            for subNiv in jerarquia[nivel]['clusters_stats'].keys():
                subNivDict={}
                '''
                #Assembler
                ass=jerarquia[nivel]['clusters_stats'][subNiv]['assembler']
                subModelName=str(ass.uid)
                subModelName=os.path.join(modelsPath,subModelName)
                ass.write().overwrite().save(subModelName)
                subNivDict.update({"assembler":subModelName})
                '''
                #Scaler
                sca=jerarquia[nivel]['clusters_stats'][subNiv]['scaler']
                subModelName="_".join(['scaler',str(nivel),str(subNiv)])
                subModelName=os.path.join(modelsPath,subModelName)
                self._store_model(sca,subModelName)
                subNivDict.update({"scaler":subModelName})

                clusterDict.update({subNiv:subNivDict})

            nivelDict.update({"clusters_stats":clusterDict})


            #print(jerarquia[nivel]['clusters_stats'])

            jerarquiaDict.update({nivel:nivelDict})  

        f=open(os.path.join(modelsPath,"taxonomy"),"w")   
        f.write(str(jerarquiaDict))
        f.close()

        
import ast

def load_taxonomy(modelsPath):
    f=open(os.path.join(modelsPath,"taxonomy"),"r")   
    strJerarquia=f.read()
    f.close()

    def loadModel(fileName):
        model=pickle.load(open(fileName, 'rb'))
        print(type(model))
        return model
       
    tJerarquia=ast.literal_eval(strJerarquia)
    tJerarquia
    resJerarquia={}
    for niv in tJerarquia.keys():
        jn=tJerarquia[niv]
        resJerarquia.update({niv:None})
        nivelDict={'vars':jn['vars']}
        '''
        #Cargar Assembler
        model=loadModel(jn['assembler'])
        nivelDict.update({'assembler':model})
        '''
        #Cargar Scaler
        model=loadModel(jn['scaler'])
        nivelDict.update({'scaler':model})
        #Cargar KMeans
        model=loadModel(jn['kmeans_model'])
        nivelDict.update({'kmeans_model':model})
        clusterDict={}
        for subNiv in jn['clusters_stats'].keys():
            subNivDict={}
            '''
            #Assembler
            ass=jn['clusters_stats'][subNiv]['assembler']
            subModel=loadModel(ass)
            subNivDict.update({"assembler":subModel})
            '''
            #Scaler
            sca=jn['clusters_stats'][subNiv]['scaler']
            subModel=loadModel(sca)        
            subNivDict.update({"scaler":subModel})

            clusterDict.update({subNiv:subNivDict})

        nivelDict.update({"clusters_stats":clusterDict})

        resJerarquia.update({niv:nivelDict})
    return resJerarquia


class covAnomalias:
    
    Pesos=None
    Anomalias=None
        
    def __init__(self, df, dft, indexCol, disclus, disEval, niveles):
        self.df = df
        self.dft = dft        
        self.indexCol = indexCol
        self.disClus = disclus
        self.disEval = disEval
        self.maxLevel = 15
        self.maxK = 15
        self.dfk_means = df 
        self._niveles = niveles
        
        #self._sc = SparkContext.getOrCreate()
        self.dft.index=self.dft[self.indexCol]

    #Vectorizar y escalar
    def _vectorize_scale(self,df_k,noVec):
        print(self.indexCol,df_k.columns)
        dataCols = list(df_k.columns)
        try:
            dataCols.remove(self.indexCol)
        except:
            a=1
        print(dataCols)
        for c in noVec:
            dataCols.remove(c)
        print("columnas",len(df_k.columns))
        print("Vectorizar DataCols", len(dataCols))
        print(dataCols)
        #assembler = VectorAssembler(inputCols=dataCols, outputCol='features')
        # dfk_vec = assembler.transform(df_k)
        dfk_vec = df_k.copy()[dataCols]
        # scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
        #                         withStd=True, withMean=True)
        scaler = StandardScaler()
        scalerModel = scaler.fit(dfk_vec)
        dfk_scal = pd.DataFrame(scalerModel.transform(dfk_vec),columns=[dataCols])
        #print("DESPUES DE ESCALAR", dfk_vec.columns)
        '''
        df_k['vecFeatures'] = dfk_scal['features']
        df_k=df_k.drop('features',axis=1)
        df_k['features'] = df_k['scaledFeatures']
        df_k=df_k.drop('scaledFeatures',axis=1)
        #self.dfk_means = df_k
        '''
        
        return dfk_scal, scalerModel

    #Aplicar el k-Means
    def _kMeans_apply(self, kMin, df):
        kmeans = KMeans(n_clusters=kMin,random_state=42) #.setDistanceMeasure(self.disClus)
        Kmeans_model = kmeans.fit(df)
        # transform your initial dataframe to include cluster assignments
        TestDataProfile = pd.DataFrame(Kmeans_model.labels_,columns=['prediction'],index=df.index)
        #TestDataProfile.select('prediction').show()
        print(TestDataProfile.groupby('prediction').size())
        print("columnas etiquetadas", len(TestDataProfile.columns))
        return Kmeans_model, TestDataProfile
    
    #CALCULAR LA IMPORTANCIA DE LAS VARIABLES CON UN RANDOM FOREST
    def _feature_importances(self, df, ass):
        df_rf = df.withColumnRenamed('prediction','label')
        #print(df_rf.columns)
        stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
        si_model = stringIndexer.fit(df_rf)
        td = si_model.transform(df_rf)
        print("columnas indexadas", len(td.columns))
        rf = RandomForestClassifier(numTrees=10, maxDepth=10,labelCol="indexed", seed=42)
        model_rf = rf.fit(td)
        print("Columnas post RF", len(td.columns))
        fi = pd.DataFrame(model_rf.featureImportances.toArray())
        fi.columns=['valor']
        #campos = pd.DataFrame(df_rf.columns[:(len(df_rf.columns))])
        campos = pd.DataFrame(ass.getInputCols())
        fi=pd.concat([fi,campos],axis=1)
        fi.columns = ['valor','variable']
        print("Campos",fi)
        fi = fi.dropna()
        fi_red = fi[fi['valor']>0.0001]
        print(fi_red.sort_values(by='valor',ascending='True'))
        if len(fi_red)>0:
            df_fi = spark.createDataFrame(fi_red)
            #df_fi = df_fi.withColumn('Level', lit(bucle))
            #df_fi = df_fi.withColumn('K', lit(kMin))
            #df_fi = df_fi.withColumn('EvalK', lit(minDist))
            df_fi_max = df_fi.agg(max(col('valor'))).first()[0]   
            #df.agg(F.max(F.abs(df.v1))).first()[0]        print(df_fi_max)
            df_fi = df_fi.withColumn('importance_norm', col('valor')/df_fi_max)
            df_fi = df_fi.sort('importance_norm',ascending=False)
            return df_fi

    def _getDummys(self,df,v):
        categ = df.select(v).distinct().rdd.flatMap(lambda x:x).collect()
        exprs = [when(col(v) == cat,1).otherwise(0)                 .alias(v+"_"+str(cat)) for cat in categ]
        df = df.select(exprs+df.columns)
        df.select(exprs).show(5)
        return df
        
    def _agglomerative_clustering(self):
        pesos = self.Pesos.select('features').toPandas()        
        series = pesos['features'].apply(lambda x : np.array(x.toArray())).as_matrix().reshape(-1,1)
        features = np.apply_along_axis(lambda x : x[0], 1, series)
        #print (features)
        clustering = AgglomerativeClustering(n_clusters=5).fit(features)
        for c in clustering.labels_:
            print(c)
        #pd.DataFrame({'Column1': clustering.labels_[:, 0]})
        clusters=pd.DataFrame(columns=['cluster'],data=clustering.labels_.flatten()) 
        print(clusters)
        pesos = tc1.Pesos.drop('features').drop('vecFeatures').toPandas()
        pesosClus = pd.concat([pesos, clusters], axis=1)
        return pesosClus
    
    def kMeansBest(self, df):
        print("ENCONTRAR MEJOR k")
        #df.show()
        minDist=0
        kMin = 0
        codo = 0
        fin = 0
        for i in range(2,self.maxK):
            #kmeans = KMeans().setK(i).setSeed(1).setDistanceMeasure(disClus)
            kmeans = KMeans(n_clusters=i,random_state=42)
            try:
                Kmeans_model = kmeans.fit(df)
                #centers=Kmeans_model.clusterCenters()
            except:
                print("No more clusters WITH ", i)
                fin=1
                break
            # transform your initial dataframe to include cluster assignments
            TestDataProfile = Kmeans_model.transform(df)
            # Evaluate clustering by computing Silhouette score
            #evaluator = ClusteringEvaluator(distanceMeasure=self.disEval)
            try:
                silhouette = silhouette_score(df, kmeans.labels_)
            except:
                print("No more clusters")
                fin=1
                break
            print("Silhouette for k="+str(i)+" with "+ self.disEval +" distances" + str(silhouette))
            if (silhouette > minDist):
                minDist = silhouette
                kMin = i
                print("k para minima distancia ="+str(i))
                codo=0
            else:
                if (silhouette <= minDist):
                    codo=codo+1
                    if codo>=3:
                        break
        #self.kMin = kMin
        return kMin,minDist, fin
    
    def _distancia(self,row,center):
        point=row[[c for c in row.index if c not in [self.indexCol,'prediction']]].to_numpy()
        dist=math.dist(point,center)
        return dist
    
    def _anomalies(self, df, kmModel):
        centers = kmModel.cluster_centers_
        print(df.dtypes)
        print(df['prediction'].drop_duplicates())
        df['distancia']=df.apply(lambda row:self._distancia(row,centers[row['prediction']]),axis=1)
        '''
        def distFromCenter(features, c, centers):
            return Vectors.dense(centers[c]).squared_distance(Vectors.dense(features))
        distancias = df.select("features", "prediction").rdd.map(lambda row: distFromCenter(row.features, row.prediction, centers)).collect()
        #Seleccionamos los puntos anómalos en función de las distancias al centroide
        import math
        #Pasamos una lista a un dataframe en spark
        #print("DISTANCIAS", distancias)
        from pyspark import SparkContext
        sc = SparkContext.getOrCreate()
        list = sc.parallelize(distancias)
        '''
        distStats=df['distancia'].describe()
        #stats = dists.stats()
        #stddev = math.sqrt(stats.variance())
        mean=distStats['mean']
        stddev = distStats['std']
        df['outliers'] = df['distancia'].apply(lambda x: ("Normal" if (math.fabs(x - mean) < 3 * stddev) else "Outlier"))
        #df = df.withColumn('Anomalia', lit(''))
        #oo = spark.createDataFrame(outliers)
        #oo.columns
         #print(outliers)
        #oo.select('0').show()
        #print('columnasoutliers',oo.columns)
        #oo.columns = ['outlier']
        #print("################################################")
        #join_on_index=self._paste_DF(df, oo)
        #print('ANOMALIAS   ')
        #join_on_index.show()
        #print("LARGO DESPUES", join_on_index.count())
        #return join_on_index.select(self.indexCol,'outliers')        
        return df
 
    def _paste_DF(self, sdf1, sdf2): 
        sdf1 = sdf1.withColumn("row_number", F.row_number().over(Window.partitionBy().orderBy(lit('A'))))
        sdf2 = sdf2.withColumn("row_number", F.row_number().over(Window.partitionBy().orderBy(lit('A'))))
        #sdf1.show()
        #sdf2.show()
        new_schema = sdf1.join(sdf2,sdf1.row_number==sdf2.row_number,'inner').drop(sdf1.row_number).drop(sdf2.row_number)
        #new_schema.show()
        return new_schema
    
    def _agrupacion_timestamp(self, df):
        colsNivel=[c for c in df.columns if c.startswith('nivel_')]
        an2=df[colsNivel].copy()
        an2['grupo']=''
        for n in range(100):
            try:
                an2["grupo"]=an2.apply(lambda row:((row['grupo']+"_") if n>0 else "") +str(row['nivel_'+str(n)]),axis=1)
            except:
                break
        print(an2.head(5))       
        print(an2.groupby(['grupo']).size().sort_index())
        df_timegrouped=an2.sort_values('grupo')
        return df_timegrouped
    
    def _anomaly_fields(self, dfk, dfa):
        '''
        dfk = dfk.join(dfa,[self.indexCol])        
        dfk = dfk.drop('features').drop('vecFeatures')
        dfk = dfk.withColumn("AnomalyFields", lit(''))
        '''
        dfk=pd.concat([dfk,dfa],axis=1).copy()
        #sc = SparkContext.getOrCreate()
        #Obtener la lista de clusters
        cluster_list = dfk['prediction'].drop_duplicates().to_list()
        for c in cluster_list:
            print(c)
            cl = dfk[dfk['prediction']==c]
            #cl.groupBy().avg().show()
            #cl.show()
            vars = list(cl.columns)
            vars.remove('prediction')
            for varN in vars:
                if 'outliers' in varN:
                    vars.remove(varN)
            vars=[c for c in vars if c!=self.indexCol]
            try:
                vars.remove('AnomalyFields')
            except:
                vars=vars
            #print("ESTAS COLUMNAS TENGO", vars)
            #print("DISTANCIAS", distancias)
            for va in vars:
                collected = gc.collect()
                print ("Garbage collector: collected %d objects." % collected)
                print(va)
                stats = cl[va].describe()
                #print (values)
                #liste = sc.parallelize(values)
                #print (liste)
                #stats = liste.stats()
                stddev = stats['std']
                print(stats)
                if stddev==0:
                    stddev=1E10
                    print("valor fijo")
                cl['value'] = cl[va].apply(lambda x: ("" if (math.fabs(x - stats['mean']) < 3 * stddev) else va+":"+str(x)))
                #print(outliers)
                #rdd = sc.parallelize(outliers)
                #schema = StructType([StructField("resp", StringType(), True)])
                #outl = spark.createDataFrame(rdd,schema)
                #outl = spark.createDataFrame(outliers,StringType())
                #outl.distinct().show()
                #cl = self._paste_DF(cl,outl)
                #print("cl     ", cl.columns)
                #cl.show()
                an4 = pd.concat([dfk,cl],axis=1)
                #print("AN3      ", an3.columns)
                an4['AnomalyFields']=an4['value'].apply(lambda x:'' if x=='' else f'{va}:{x}')
                cl = cl.drop('value',axis=1)
                
            #Cada cluster tiene un numero diferente de filas y luego hay que unirlas todas
            try:
                dfk_total = dfk_total.unionAll(an4)
            except:
                dfk_total = an4                          
                
        return dfk_total
    
    def _distancia(self,row,center):
        try:
            point=row[[c for c in row.index if c not in [self.indexCol,'prediction']]].to_numpy()
        except:
            point=row
        dist=math.dist(point,center)
        return dist
      
    def jerarquia(self):
        fin=0
        dfk_Original=self.dfk_means
        disClus, disEval = self.disClus, self.disEval
        for niv in self._niveles['Nivel'].unique():
            collected = gc.collect()
            print ("Garbage collector: collected %d objects." % collected)
            sNiv = str(niv).replace('.0','')
            print("NIVEL ",sNiv)
            varNivel = self._niveles[self._niveles['Nivel']==niv]['variable']
            print(varNivel)
            lisVar=[self.indexCol]
            for var in varNivel:
                lisVar=lisVar+[var]
            df_kVar = self.dfk_means[lisVar]
            print(df_kVar.columns)            
            df_kVar_Vec, scal = self._vectorize_scale (df_kVar, '')
            #self.dfk_means.show()
            k, dist, self.fin = self.kMeansBest(df_kVar_Vec)
            #Aplicar el k-Means
            km_model, dfk_labeled = self._kMeans_apply(k,df_kVar_Vec)
            print('DFK_LABELED ANTES DE TOCAR', dfk_labeled.dtypes)
            #dfk_labeled.show()
            newCol = 'nivel_'+sNiv
            print (newCol)
            dfk_cluster = dfk_labeled.copy()  #['prediction']
            dfk_cluster.columns=[newCol]
            dfk_cluster.index=df_kVar.index
            #dfk_cluster.show()
            #print("KKK ", k)
            dfk_labeled.index=df_kVar.index
            dfk_labeled=pd.concat([df_kVar,dfk_labeled],axis=1)            
            #dfk_labeled.select('prediction').show(10)
            #print(dfk_labeled.dtypes)
            ols = self._anomalies(dfk_labeled, km_model)
            #print("ols",ols)
            ols = ols[[self.indexCol,'outliers']]
            #ols.groupby('outliers').count().show()
            ols.columns=[self.indexCol,'outliers_'+sNiv.replace('.0','')]
            #ols.show()
            try:
                self.Anomalias = pd.concat([self.Anomalias,ols],axis=1)
            except:
                self.Anomalias = ols
            self.Anomalias = pd.concat([self.Anomalias,dfk_cluster], axis=1)
            collected = gc.collect()
            print ("Garbage collector: collected %d objects." % collected)
            
            print("DETECTAR ORIGEN ANOMALIA")
            df_af=self._anomaly_fields(dfk_labeled,ols)
            df_af['fieldsAnomaly_Nivel_'+sNiv]=df_af['AnomalyFields']
            df_af.drop('AnomalyFields',axis=1)
            try:
                self.CamposAnomalos = pd.concat([self.CamposAnomalos,df_af],axis=1)
            except:
                self.CamposAnomalos = df_af
            print("RECOPILACION CAMPOS ANOMALOS")
            #print(self.CamposAnomalos.count())
            
            print(self.CamposAnomalos['fieldsAnomaly_Nivel_'+sNiv].drop_duplicates())
        
        print("FIN ANOMALIAS, CAMPOS RESPONSABLES")
        print(self.CamposAnomalos.columns)        
        selected = [s for s in self.CamposAnomalos.columns if 'fieldsAnomaly' in s]+[s for s in self.CamposAnomalos.columns if 'outlier' in s]
        print(self.CamposAnomalos[selected].drop_duplicates())
        
        collected = gc.collect()
        print ("Garbage collector: collected %d objects." % collected)

        #return
    
        df_grouped = self._agrupacion_timestamp(self.Anomalias)
        print("AGRUPADO")
        print(df_grouped)
        colsNivel=df_grouped.columns
        # colsNivel.remove(self.indexCol)
        #print(df_grouped.groupby(colsNivel).size().sort_values())
        
        dfNiveles=pd.concat([self.dft,df_grouped],axis=1)
        #dfNiveles.show()
        try:
            dfNiveles=dfNiveles.drop(self.indexCol,axis=1)
        except:
            dfNiveles=dfNiveles
            
        dfNiveles=dfNiveles.drop([c for c in dfNiveles.columns if c.startswith('nivel_')],axis=1)
        seq=dfNiveles.pivot_table(index="timestamp",columns='grupo',aggfunc='size',fill_value=0.0)
        #.pivot_table("grupo").count().sort('timestamp')
        #seq=seq.fillna(0)
        
        self.seq = seq
        
        # seq.show()

        centers = seq.mean()
        print(centers)

        centersA = np.array(centers)
        #print(centersA)

        #centers['timestamp']=''

        '''
        preCenters = centers
        for name in preCenters.columns:
          preCenters = preCenters.withColumnRenamed(name, name.replace('avg(', '').replace(')',''))
        preCenters = preCenters[seq.columns]
        '''
        #seq = seq.unionAll(preCenters)
        #seq=pd.concat([seq,pd.DataFrame([centers])])
        print("ANOMALIAS POR GRUPOS DE TIEMPO Y CLUSTERS")
        
        dataCols = seq.columns
        try:
            dataCols.remove('timestamp')
        except:
            a=1
        print("columnas",len(seq.columns))
        print("Vectorizar DataCols", len(dataCols))
        print(dataCols)
        #assembler = VectorAssembler(inputCols=dataCols, outputCol='features')
        #dfk_vec = assembler.transform(seq)
        dfk_vec  = seq
        scaler = StandardScaler()
        scalerModel = scaler.fit(dfk_vec)
        dfk_scal = scalerModel.transform(dfk_vec)
        #print("DESPUES DE ESCALAR", dfk_vec.columns)
        #df_k0 = dfk_scal.withColumn('vecFeatures', col('features')).drop('features')
        #df_k0 = df_k0.withColumn('features', col('scaledFeatures')).drop('scaledFeatures')
        #self.dfk_means = df_k

        #dfk_vec = df_k0.withColumn('prediction',lit(0.0))

        # dfk_vec.dtypes
        '''
        #centers = np.nparray(seq.groupBy().avg())
        def distFromCenter(features, centers):
            return Vectors.dense(centers).squared_distance(Vectors.dense(features))
        distancias = dfk_vec["vecFeatures", "prediction"].rdd.map(lambda row: distFromCenter(row.vecFeatures, centersA[0])).collect()
        '''
        distancias=[self._distancia(row,centers) for row in dfk_scal] 
        '''
        #Seleccionamos los puntos anómalos en función de las distancias al centroide
        import math
        #Pasamos una lista a un dataframe en spark
        #print("DISTANCIAS", distancias)
        from pyspark import SparkContext
        sc = SparkContext.getOrCreate()
        list = sc.parallelize(distancias)
        '''
        dfDist=pd.DataFrame(distancias,columns=['distancia'])
        stats = dfDist['distancia'].describe()
        stddev = stats['std']
        dfDist['outliers'] = dfDist['distancia'].apply(lambda x: ("Normal" if (math.fabs(x - stats['mean']) < 3 * stddev) else "Outlier"))
        #df = df.withColumn('Anomalia', lit(''))
        #oo = spark.createDataFrame(outliers)
        '''
        oo  = spark.createDataFrame(pd.DataFrame(outliers))
        oo.columns
        
        oo.show()
        '''
        dfDist.index=self.seq.index
        self.seq=pd.concat([self.seq,dfDist],axis=1)        
        
        



    
# In[4]:

#Datos de Ibermática

input_data="processed_data_eth_ip_tcp_opcua_total.csv"
modelsPath=r"/Modelos"

##### Entrada 

df = pd.read_csv(input_data, sep=";")
print('Filas: '+ str(len (df))+ '  Columnas: '+ str(len(df.columns))) #number columns: max number of fields in the mongo collection


for c in df.columns:
    if '_Timestamp' in c:
        df = df.drop([c],axis=1)
print('Filas: '+ str(len (df))+ '  Columnas: '+ str(len(df.columns))) #number columns: max number of fields in the mongo collection


# ## Limpieza de datos
# ### Eliminar ids y fechas/horas
#'_Timestamp', '_id' variables not necessary for analytics. 
for c in df.columns:
    if c.endswith('_id'):
        df = df.drop([c], axis=1)
    if c.endswith('_Timestamp'):
        df = df.drop([c], axis=1)
    if c.endswith('_timestamp'):
        df = df.drop([c], axis=1)
len(df.columns)


# #### Tratamiento de valores nulos
# Como se puede observar en la tabla 'summary_total', hay dos variables que no están al 100% completas: PGG_SELECTED_VALUES_layers_opcua_text_text_10, PGG_SELECTED_VALUES_layers_opcua_text_text_9. Imputamos un EL VALOR 'N'.
# #### Variables no relevantes para la analitica

no_relevantes = ['PGG_DATE_INSERT','PGG_PROCESS','PGG_DB_NAME','PGG_PATTERN_TYPE','PGG_TRANSFORMATION_TYPE']
df_r = df.drop(no_relevantes, axis=1)

print(len(df_r['PGG_HASH_INDEX'].unique()))
indexCol = 'PGG_HASH_INDEX'

# #### Tomar los datos transformados para KMeans
selected = [s for s in df.columns if 'PGG_TRANSFORM_' in s] 
selected += [indexCol]
df_k = df_r[selected]


# ## Cargamos los datos

dataCols = list(df_k.columns)
dataCols.remove(indexCol)
df_k = fill_with_xtrem(df_k,dataCols)



# #### Unificar tipos
# 
# Para poder vectorizar hay que convertir todos los datos al mismo tipo, en este caso Double
df_k = df_k.fillna(0.0)



df_k[dataCols]=df_k[dataCols].astype('float')
print(df_k.dtypes)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit(df_k[dataCols]))
print(scaler.mean_)
df_ks=scaler.transform(df_k[dataCols])


df_ks=pd.DataFrame(df_ks)


for c in df_k.columns:
    newC=c.replace('PGG_TRANSFORM_PGG_SELECTED_VALUES_layers_','')
    if newC!=c:
        df_k[newC]=df_k[c]
        df_k=df_k.drop([c],axis=1)

df_k.columns


# ### Taxonomía con covarianza
# ##### Remove useless attributes
# ##### Remove correlated attributes
# ##### Rellenar missing values
# #Con  nulo o zero
# #Esto no es lo que habíamos quedado
# 
# #### Covarianza Taxonomy


df_k.columns
tc1 = covTaxonomy(df_k,'PGG_HASH_INDEX','euclidean','squaredEuclidean')
tc1.jerarquia()


tc1.Niveles
print(tc1.modelosKMeans)
tc1.store(modelsPath)
redecTaxonomy=load_taxonomy(modelsPath)


import gc

collected = gc.collect()
print ("Garbage collector: collected %d objects." % collected)


tc1.Niveles


an1 = covAnomalias(df_k,df[['PGG_HASH_INDEX','timestamp']],'PGG_HASH_INDEX','euclidean','squaredEuclidean',tc1.Niveles)
an1.jerarquia()


co = an1.CamposAnomalos.columns
camposAnomalia = []
for c1 in co:
    if 'Nivel_' in c1:
        camposAnomalia = camposAnomalia + [c1]        
    if 'outliers_' in c1:
        camposAnomalia = camposAnomalia + [c1]        
print(an1.CamposAnomalos[camposAnomalia].drop_duplicates())



an1.CamposAnomalos.to_csv("anomalias.1.csv")



#%% Aplicación del Código


def clasifica(self, df, niveles):
    nivList=[]
    dfk_Labeled=df
    print("COLS INICIAL", df.columns)
    for n in niveles.keys():
        if type(n)==str: continue
        n=int(n)
        print("NIVEL ", n)
        #print(self._niveles[n])
        selectedCols=niveles[n]['vars']
        print("VARIABLES ", niveles[n]['vars'])
        scaler=niveles[n]['scaler']
        model=niveles[n]['kmeans_model']
        dfVec=dfk_Labeled.copy()[selectedCols]
        print("COLUMNAS VEC ",dfVec.columns)
        dfScaled=pd.DataFrame(scaler.transform(dfVec),columns=dfVec.columns)
        print("COLUMNAS SCAL ",dfScaled)
        print("Scaled")
        clusters=model.predict(dfScaled)
        dfk_Labeled['nivel_'+str(n)]=clusters
        colNivel='nivel_'+str(n)
        nivList.append(colNivel)
        print("LABELED", dfk_Labeled.columns)
        dfk_Labeled['group']=dfk_Labeled[nivList].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        print(dfk_Labeled)
    return dfk_Labeled


clas = clasifica(1,df_k,tc1.modelosKMeans)




