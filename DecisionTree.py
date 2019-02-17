import sys
import pandas as pd
import numpy as np
import copy
import math
from pandas.api.types import is_numeric_dtype
import numbers
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings("ignore")

class DecisionTree():
    
    def __init__(self, method="ID3", measure="entropy", max_depth=5):
        self.root = None
        self.method = method
        self.measure = measure
        self.max_depth = max_depth
        self.cnt=1

    def fit(self, X, y):
        self.root= self.createTree(X, y, list(X.columns.values), self.max_depth)

    def predict(self, X):
        y = []
        for index, row in X.iterrows():
            y.append(self.findTree(self.root, row))
        return y

    def findTree(self, node, X):
        if node.isLeaf==True:
            return node.label
        else:
            f = X[node.feature]
            if node.split==None:
                return self.findTree(node.branch[f], X)
            else:
                if f < node.split:
                    return self.findTree(node.branch["<"], X)
                else:
                    return self.findTree(node.branch[">="], X)

    def exportTree(self):
        self.showTree(self.root, 0)
    

    def showTree(self, node, indent):
        if node.isLeaf==True:
         
            print("    "*indent+"===> "+str(node.label))
            return
        
        print("    "*indent +"--- "+ "[" + str(node.feature) + "]")
        indent += 1        
        for condition in node.branch:
            if node.split==None:
                print("    "*indent +"--- "+ str(condition))
            else:
                print("    "*indent +"--- "+ str(condition) + str(node.split))
            self.showTree(node.branch[condition], indent+1)

    def createTree(self, X, y, attribs, depth_now):
        if self.method=="C4.5":
            return self.C4dot5(X, y, attribs, depth_now)
        else:
            return self.ID3(X, y, attribs, depth_now)
    
    def ID3(self, X, y, attribs, depth_now):
        root = TreeNode()
        unique_vals, counts = np.unique(y.values, return_counts=True)
        if len(unique_vals) == 1:
            root.setLabel(unique_vals[0])
        elif len(attribs)==0 or depth_now == 0:
            root.setLabel(unique_vals[np.argmax(counts)])
        else:
            attr, split_val = self.findBestAttribute(X, y, attribs)
            new_attribs = copy.deepcopy(attribs)
            new_attribs.remove(attr)
            root.setFeature(attr)
            unique_vals = np.unique(X[attr].values)
            for val in unique_vals:
                root.addBranch(self.ID3(X[X[attr] == val], y[X[attr] == val], new_attribs, depth_now-1), val)
        return root

    def C4dot5(self, X, y, attribs, depth_now):
        root = TreeNode()
        unique_vals, counts = np.unique(y.values, return_counts=True)
        if len(unique_vals) == 1:
            root.setLabel(unique_vals[0])
        elif len(attribs)==0 or depth_now == 0:
            root.setLabel(unique_vals[np.argmax(counts)])
        else:
            attr, split_val = self.findBestAttribute(X, y, attribs)
            #print(attr, split_val)
            if attr == None:
                root.setLabel(unique_vals[np.argmax(counts)])
            else:
                new_attribs = copy.deepcopy(attribs)
                new_attribs.remove(attr)
                root.setFeature(attr)
                if split_val==None:
                    unique_vals = np.unique(X[attr].values)
                    for val in unique_vals:
                        root.addBranch(self.C4dot5(X[X[attr] == val], y[X[attr] == val], new_attribs, depth_now-1), val)
                        self.cnt+=1
                else:
                    Z = X[attr].values < split_val
                    root.addBranch(self.C4dot5(X[Z], y[Z], new_attribs, depth_now-1), "<", split_val)
                    self.cnt+=1
                    root.addBranch(self.C4dot5(X[~Z], y[~Z], new_attribs, depth_now-1), ">=", split_val)
                    self.cnt+=1
        return root


    def findBestAttribute(self, X, y, attribs):
        optimal_attr = None
        optimal_split = None
        maxIG = 0
        for attr in attribs:
            #print("[["+attr+"]]")
            IG, split_val = self.gain(X, y, attr)
            #print(IG)
            if optimal_attr == None or IG > maxIG:
                optimal_attr = attr
                optimal_split = split_val
                maxIG = IG
        if maxIG == 0:
            optimal_attr = None
            optimal_split = None
        #print("-------->"+optimal_attr+"\n")
        return optimal_attr, optimal_split

    
    def gain(self, X, y, attr):
        if self.method=="C4.5":
            if is_numeric_dtype(X[attr]):
                IG, split_val = self.findBestSplit(X, y, attr)
            else:
                IG = gain_ratio(X, y, attr,None,self.measure)
                #IG = information_gain(X, y, attr)
                split_val = None
        else:
            IG = information_gain(X, y, attr,self.measure)
            split_val = None
        return IG, split_val

    def findBestSplit(self, X, y, attr):
        valList = np.sort(X[attr].values)

        optimal_split = None
        maxIG = 0
        for idx in range(len(valList)-1):
            if valList[idx]==valList[idx+1]:
                continue
            split_val = float(valList[idx] + valList[idx+1])/2
            #print("split at:", split_val)
            IG = gain_ratio(X, y, attr, split_val,self.measure)
            #print("= ", IG)
            if optimal_split == None or IG > maxIG:
                optimal_split = split_val
                maxIG = IG    
            
        #print("--->", maxIG, optimal_split)
        return maxIG, optimal_split



def entropy(y):
    n = y.shape[0]
    unique_vals, counts = np.unique(y.values, return_counts=True)
    e = 0
    for nt in counts:  
        prob = float(nt/n)
        e += -prob * math.log(prob, 2)
    #print("   +"+str(e))
    return e


def gini(y):
    n = y.shape[0]
    unique_vals, counts = np.unique(y.values, return_counts=True)
    e = 0
    for nt in counts:  
        prob = float(nt/n)
        e += (prob*prob)
    #print("   +"+str(e))
    return (1-e)


def misclass(y):
    n = y.shape[0]
    unique_vals, counts = np.unique(y.values, return_counts=True)
    e = 0
    maxprob=-1
    for nt in counts:  
        prob = float(nt/n)
        if(prob>maxprob):
            maxprob=prob
    #print("   +"+str(e))
    return (1-maxprob)



def information_gain(X, y, attr,measure):
    if measure=="entropy":
        unique_vals, counts = np.unique(X[attr].values, return_counts=True)
        #print("IG: ")
        IG = entropy(y)
        n = X.shape[0]
        for (val, nt) in zip(unique_vals, counts):        
            #print(val)
            child = y[X[attr] == val]
            #print("(*"+str(nt)+"/"+str(n)+")")
            IG -= (nt/n) * entropy(child)
        #print("  = "+str(IG))
        return IG
    elif measure=="gini":
        unique_vals, counts = np.unique(X[attr].values, return_counts=True)
        #print("IG: ")
        IG = gini(y)
        n = X.shape[0]
        for (val, nt) in zip(unique_vals, counts):        
            #print(val)
            child = y[X[attr] == val]
            #print("(*"+str(nt)+"/"+str(n)+")")
            IG -= (nt/n) * gini(child)
        #print("  = "+str(IG))
        return IG
    else:
        unique_vals, counts = np.unique(X[attr].values, return_counts=True)
        #print("IG: ")
        IG = misclass(y)
        n = X.shape[0]
        for (val, nt) in zip(unique_vals, counts):        
            #print(val)
            child = y[X[attr] == val]
            #print("(*"+str(nt)+"/"+str(n)+")")
            IG -= (nt/n) * misclass(child)
        #print("  = "+str(IG))
        return IG


def gain_ratio(X, y, attr, split_val = None,measure=""):
    if measure=="entropy":
        if split_val==None:
            unique_vals, counts = np.unique(X[attr].values, return_counts=True)   
            IG = entropy(y)
            n = X.shape[0]
            for (val, nt) in zip(unique_vals, counts):      
                child = y[X[attr] == val]
                IG -= (nt/n) * entropy(child)
        else:
            Z = X[attr].values < split_val
            unique_vals, counts = [True, False], [X[Z].shape[0], X[~Z].shape[0]]
            if 0 in counts: 
                counts.remove(0)
            IG = entropy(y)
            #print("ori:"+str(IG))
            n = X.shape[0]
            for (val, nt) in zip(unique_vals, counts):      
                if val==True: child = y[Z]
                else: child = y[~Z]
                #print(child)
                #print("(*"+str(nt)+"/"+str(n)+")")
                IG -= (nt/n) * entropy(child)
            #print("="+str(IG))
        return IG
        split_info = 0
        maxprob=-1
        for nt in counts:  
            prob = float(nt/n)
            split_info += -prob * math.log(prob, 2)
            #if(prob>maxprob):
            #    maxprob=prob
            #split_info +=prob*prob
        #split_info=1-split_info
        #split_info=1-maxprob
        if len(counts) == 1:
            return IG
        else:
            #print(IG, "/", split_info)
            return IG/split_info

    elif measure=="gini":
        if split_val==None:
            unique_vals, counts = np.unique(X[attr].values, return_counts=True)   
            IG = gini(y)
            n = X.shape[0]
            for (val, nt) in zip(unique_vals, counts):      
                child = y[X[attr] == val]
                IG -= (nt/n) * gini(child)
        else:
            Z = X[attr].values < split_val
            unique_vals, counts = [True, False], [X[Z].shape[0], X[~Z].shape[0]]
            if 0 in counts: 
                counts.remove(0)
            IG = gini(y)
            #print("ori:"+str(IG))
            n = X.shape[0]
            for (val, nt) in zip(unique_vals, counts):      
                if val==True: child = y[Z]
                else: child = y[~Z]
                #print(child)
                #print("(*"+str(nt)+"/"+str(n)+")")
                IG -= (nt/n) * gini(child)
            #print("="+str(IG))
        return IG
        split_info = 0
        maxprob=-1
        for nt in counts:  
            prob = float(nt/n)
            #split_info += -prob * math.log(prob, 2)
            #if(prob>maxprob):
            #    maxprob=prob
            split_info +=prob*prob
        split_info=1-split_info
        #split_info=1-maxprob
        if len(counts) == 1:
            return IG
        else:
            #print(IG, "/", split_info)
            return IG/split_info

    elif measure=="misclass":
        if split_val==None:
            unique_vals, counts = np.unique(X[attr].values, return_counts=True)   
            IG = misclass(y)
            n = X.shape[0]
            for (val, nt) in zip(unique_vals, counts):      
                child = y[X[attr] == val]
                IG -= (nt/n) * misclass(child)
        else:
            Z = X[attr].values < split_val
            unique_vals, counts = [True, False], [X[Z].shape[0], X[~Z].shape[0]]
            if 0 in counts: 
                counts.remove(0)
            IG = misclass(y)
            #print("ori:"+str(IG))
            n = X.shape[0]
            for (val, nt) in zip(unique_vals, counts):      
                if val==True: child = y[Z]
                else: child = y[~Z]
                #print(child)
                #print("(*"+str(nt)+"/"+str(n)+")")
                IG -= (nt/n) * misclass(child)
            #print("="+str(IG))
        return IG
        split_info = 0
        maxprob=-1
        for nt in counts:  
            prob = float(nt/n)
            split_info += -prob * math.log(prob, 2)
            if(prob>maxprob):
                maxprob=prob
            #split_info +=prob*prob
        #split_info=1-split_info
        split_info=1-maxprob
        if len(counts) == 1:
            return IG
        else:
            #print(IG, "/", split_info)
            return IG/split_info

class TreeNode():
    def __init__(self):
        self.feature = ""
        self.label = ""
        self.isLeaf = False
        self.branch = {}
        self.split = None

    def setFeature(self, feature):
        self.feature = feature

    def setLabel(self, label):
        self.label = label
        self.isLeaf = True

    def addBranch(self, child, condition, split=None):
        self.split = split
        self.branch[condition] = child
    

if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]

    #df = pd.read_csv(inputPath, header=0, index_col=False)
    df = pd.read_csv(train)
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('left')))
    df1 = df.ix[:, cols]
    df2 = df1.replace({'salary' : { 'low': 1, 'medium': 2, 'high' :3}})
    df3 = df2.replace({'sales' : {'sales' : 1, 'technical' : 2, 'support' :3, 'IT' :4, 
                                'product_mng' : 5, 'marketing' : 6, 'RandD' :7, 
                                'accounting' :8, 'hr' : 9, 'management' : 10}})
    df3["promotion_last_5years"]=df3["promotion_last_5years"].astype(str)
    df3["Work_accident"]=df3["Work_accident"].astype(str)
    df3["number_project"] = df3["number_project"].astype(str)
    X=df3.iloc[:,1:10]
    y=df3.iloc[:,0]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.20,random_state=1200)
    dt = DecisionTree(method="C4.5", measure="entropy",max_depth=5)
    dtree1=dt.fit(X_train, y_train)
    y_pred_train=dt.predict(X_train)
    accuracy_mdl=accuracy_score(y_train,y_pred_train)
    conf_matrix=confusion_matrix(y_train,y_pred_train)
    print('Training accuracy score :',round(accuracy_mdl*100,2),'%')
    y_pred = dt.predict(X_test)
    accuracy_mdl=accuracy_score(y_test,y_pred)
    conf_matrix=confusion_matrix(y_test,y_pred)
    print('Validation accuracy score :',round(accuracy_mdl*100,2),'%')
    print(conf_matrix)
    print(classification_report(y_test,y_pred))
    
    df = pd.read_csv(test)
    cols = list(df)
    cols.insert(0, cols.pop(cols.index('left')))
    df1 = df.ix[:, cols]
    df2 = df1.replace({'salary' : { 'low': 1, 'medium': 2, 'high' :3}})
    df3 = df2.replace({'sales' : {'sales' : 1, 'technical' : 2, 'support' :3, 'IT' :4, 
                                'product_mng' : 5, 'marketing' : 6, 'RandD' :7, 
                                'accounting' :8, 'hr' : 9, 'management' : 10}})
    df3["promotion_last_5years"]=df3["promotion_last_5years"].astype(str)
    df3["Work_accident"]=df3["Work_accident"].astype(str)
    df3["number_project"] = df3["number_project"].astype(str)
    X=df3.iloc[:,1:10]
    y=df3.iloc[:,0]
    y_pred = dt.predict(X)
    accuracy_mdl=accuracy_score(y,y_pred)
    conf_matrix=confusion_matrix(y,y_pred)
    print('Testing accuracy score :',round(accuracy_mdl*100,2),'%')
    print(conf_matrix)
    print(classification_report(y,y_pred))
