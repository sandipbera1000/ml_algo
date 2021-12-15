# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 19:12:38 2021

@author: Sandip
"""


import pandas as pd
kdd=pd.read_csv("kddcup99_csv.csv",nrows=100000)
#print(kdd)

import numpy as np
import pandas as pd
from sklearn import preprocessing


loguni= kdd['logged_in'].unique()

a=kdd['label']

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
#Encode labels in column 'label'.
kdd['label']= label_encoder.fit_transform(kdd['label'])
 
labeled=kdd['label']
kdd['label']


X = kdd.iloc[:,[4,5,10,11,23,24]].values
Y = labeled
m= kdd.iloc[:,[4,5,10,11,23,24]]
#uni= Y.unique()

#pd.set_option('display.max_columns', None)
#p=pd.DataFrame([a,Y])

d=Y.unique()

t= a.unique()

pd.set_option('display.max_columns', None)
q=pd.DataFrame([d,t])

print("after level encoding Which Numeric Value Represent Which Attack:")

print("****************************************************************")
print(q)
print("****************************************************************")

#################################################################
#              fitting data into dessisionTree                  #
#################################################################



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classifier to the Training set

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini',
                                    random_state = 42)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#acuracy printing

from sklearn.metrics import accuracy_score
print("Accuracy of DecissionTree:",accuracy_score(y_test, y_pred))

DTA = accuracy_score(y_test, y_pred)

'''
pr=sc.transform([[1000,5,0]])
predict_result=classifier.predict(pr)

print("DTA predict",predict_result)

'''
#################################################################
#              fitting data into support Vector Machine         #
#################################################################




from sklearn.preprocessing import StandardScaler



#sc=StandardScaler()
#X=sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, Y, 
                            test_size=0.25, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
  
from sklearn.svm import SVC
#Linear SVM Classification
classifier = SVC(kernel = 'linear', random_state = 42)

#training the support vector machine model
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy of SVM: ",accuracy_score(y_test, y_pred))
SVM = accuracy_score(y_test, y_pred)


'''
pr=sc.transform([[400,0,1]])
predict_result=classifier.predict(pr)

print("SVM Predict:",predict_result)
'''




#################################################################
#              fitting data inti      RandomForestClassifier    #
#################################################################



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 900, 
                criterion = 'gini', random_state = 42)


classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy of  RandomForest: ",accuracy_score(y_test, y_pred))
RND = accuracy_score(y_test, y_pred)



'''
pr=sc.transform([[1000,0,1]])
predict_result=classifier.predict(pr)
print("Random Forest Prdict :",predict_result)

'''
print()

#################################################################
#              finding best algorithm for the data set          #
#################################################################

if DTA > SVM and DTA > RND:
    print("DecisionTree IS BEST FIT ALGORITHM FOR This DATA SET")
    
elif SVM > DTA and SVM > RND:
    print("Support Vector Machine is BEST FIT ALGORITHM FOR This DATA SET")
    
elif RND > DTA and RND > SVM:
    print("RandomForest is BEST FIT ALGORITHM FOR This DATA SET")
    
    

#################################################################
#             Predicting Attack Type With RandomForest          #
#################################################################


print()

    
'''   
while True:
    
    
    
    print("1. Enter Values To Predict with random forest : ")
    
    print("2. exit ")
    
    inp=int(input("Enter Your Choice: "))
    
    if inp == 1:
        
        
        
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
       
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 900, criterion = 'gini', random_state = 42)
        classifier.fit(X_train, y_train)
        y_pred=classifier.predict(X_test)
        
        
        
        
        pd.set_option('display.max_columns', None)
        prd=pd.DataFrame([a,y_pred])
        
        print(prd)
        
        #acuracy printing
        print()
        print()
        from sklearn.metrics import accuracy_score
        print("Accuracy of  RandomForest: ",accuracy_score(y_test, y_pred))
        print()
        print()
        
        
        source= int(input("Enter Source byte: "))
        dst=int(input("Enter destnation byte: "))
        logf=int(input("Enter How Many Time Login Failed: "))
        login=int(input("Enter '0'(for not logged in ) Enter '1' (for loged in)  : "))
        serv_count=int(input("Enter Server Count: "))
        serror=int(input("Enter Serror: "))
        
        p=sc.transform([[source,dst,logf,login,serv_count,serror]])
        predict_result = classifier.predict(p)
        print('After Predicttion with random forest : ',predict_result)
        
        
        if predict_result == 0:
            print("Attack Type is buffer_overflow")
            print()
        elif predict_result == 1:
            print("Attack Type is guess_passwd ")
            print()
        elif predict_result == 2:
            print("Attack Type is loadmodule ")
            print()
        elif predict_result == 3:
            print("Attack Type is neptune ")
            print()
        elif predict_result == 4:
            print("Normal")
            print()
        elif predict_result == 5:
            print("Attack Type is perl")
            print()
        elif predict_result == 6:
            print("Attack Type is Pod")
            print()
        elif predict_result == 7:
            print("Attack Type is smurf")
            print()
        elif predict_result == 8:
            print("Attack Type is teardrop ")
            print()
        else:
            print("It is a New Type Of Attack")
            print()
    
    
    elif inp == 2:
       break
            
 '''

while True:
    
    
    
    print("1. Enter Values To Predict With RandomForest: ")
    print("2. Enter Values To Predict with SVM: ")
    print("3. Enter Values To Predict With DecissionTree: ")
    
    
    
    print("0. exit ")
    
    inp=int(input("Enter Your Choice: "))
    
    if inp == 1:
        
        
        
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
       
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 900, criterion = 'gini', random_state = 42)
        classifier.fit(X_train, y_train)
        y_pred=classifier.predict(X_test)
        
        
        
        '''
        pd.set_option('display.max_columns', None)
        prd=pd.DataFrame([a,y_pred])
        
        print(prd)
        '''
        #acuracy printing
        print()
        print()
        from sklearn.metrics import accuracy_score
        print("Accuracy of  RandomForest: ",accuracy_score(y_test, y_pred))
        print()
        print()
        
        
        source= int(input("Enter Source byte: "))
        dst=int(input("Enter destnation byte: "))
        logf=int(input("Enter How Many Time Login Failed: "))
        login=int(input("Enter '0'(for not logged in ) Enter '1' (for loged in)  : "))
        serv_count=int(input("Enter Server Count: "))
        serror=int(input("Enter Serror: "))
        p=sc.transform([[source,dst,logf,login,serv_count,serror]])
        predict_result = classifier.predict(p)
        print('After Predicttion with random forest : ',predict_result)
        
        
        if predict_result == 0:
            print("Attack Type is buffer_overflow")
            print()
        elif predict_result == 1:
            print("Attack Type is guess_passwd ")
            print()
        elif predict_result == 2:
            print("Attack Type is loadmodule ")
            print()
        elif predict_result == 3:
            print("Attack Type is neptune ")
            print()
        elif predict_result == 4:
            print("Normal")
            print()
        elif predict_result == 5:
            print("Attack Type is perl")
            print()
        elif predict_result == 6:
            print("Attack Type is Pod")
            print()
        elif predict_result == 7:
            print("Attack Type is smurf")
            print()
        elif predict_result == 8:
            print("Attack Type is teardrop ")
            print()
        else:
            print("wrong preditction")
            print()
    elif inp == 2:
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test= train_test_split(X, Y, 
                            test_size=0.25, random_state=42)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)


        from sklearn.svm import SVC
        #Linear SVM Classification
        classifier = SVC(kernel = 'linear', random_state = 0)

        #training the support vector machine model
        classifier.fit(X_train, y_train)
        y_pred=classifier.predict(X_test)
        from sklearn.metrics import accuracy_score
        print("Accuracy: ",accuracy_score(y_test, y_pred))
        #SVM=accuracy_score(y_test, y_pred)
        
        source= int(input("Enter Source byte: "))
        dst=int(input("Enter destnation byte: "))
        logf=int(input("Enter How Many Time Login Failed: "))
        login=int(input("Enter '0'(for not logged in ) Enter '1' (for loged in)  : "))
        serv_count=int(input("Enter Server Count: "))
        serror=int(input("Enter Serror: "))
        p=sc.transform([[source,dst,logf,login,serv_count,serror]])
        predict_result = classifier.predict(p)
        print('After Predicttion with SVM : ',predict_result)
        
        if predict_result == 0:
            print("Attack Type is buffer_overflow")
            print()
        elif predict_result == 1:
            print("Attack Type is guess_passwd ")
            print()
        elif predict_result == 2:
            print("Attack Type is loadmodule ")
            print()
        elif predict_result == 3:
            print("Attack Type is neptune ")
            print()
        elif predict_result == 4:
            print("Normal")
            print()
        elif predict_result == 5:
            print("Attack Type is perl")
            print()
        elif predict_result == 6:
            print("Attack Type is Pod")
            print()
        elif predict_result == 7:
            print("Attack Type is smurf")
            print()
        elif predict_result == 8:
            print("Attack Type is teardrop ")
            print()
        else:
            print("wrong preditction")
            print()
        
    elif inp == 3:

        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting Decision Tree Classifier to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'gini',
                                    random_state = 42)
        classifier.fit(X_train, y_train)
        y_pred=classifier.predict(X_test)
        #acuracy printing
        from sklearn.metrics import accuracy_score
        print("Accuracy:",accuracy_score(y_test, y_pred))
        #DTA=accuracy_score(y_test, y_pred)
        
        source= int(input("Enter Source byte: "))
        dst=int(input("Enter destnation byte: "))
        logf=int(input("Enter How Many Time Login Failed: "))
        login=int(input("Enter '0'(for not logged in ) Enter '1' (for loged in)  : "))
        serv_count=int(input("Enter Server Count: "))
        serror=int(input("Enter Serror: "))
        p=sc.transform([[source,dst,logf,login,serv_count,serror]])
        predict_result = classifier.predict(p)
        print('After Predicttion with Decission Tree : ',predict_result)
        
        if predict_result == 0:
            print("Attack Type is buffer_overflow")
            print()
        elif predict_result == 1:
            print("Attack Type is guess_passwd ")
            print()
        elif predict_result == 2:
            print("Attack Type is loadmodule ")
            print()
        elif predict_result == 3:
            print("Attack Type is neptune ")
            print()
        elif predict_result == 4:
            print("Normal")
            print()
        elif predict_result == 5:
            print("Attack Type is perl")
            print()
        elif predict_result == 6:
            print("Attack Type is Pod")
            print()
        elif predict_result == 7:
            print("Attack Type is smurf")
            print()
        elif predict_result == 8:
            print("Attack Type is teardrop ")
            print()
        else:
            print("wrong preditction")
            print()
            
    elif inp == 0:
        break
              
            
            
            
        
            
            
        
            
            
            
        
            
    
    





























