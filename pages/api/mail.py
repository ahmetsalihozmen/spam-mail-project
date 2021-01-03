import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
import heapq
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical 
import os
import sys








datasets={'./pages/api/messages.csv':1, './pages/api/spam.csv':2, './pages/api/emails.csv':3, './pages/api/spamlar.csv':4}

if(datasets[sys.argv[1]]==1):
        data=pd.read_csv(sys.argv[1])
   
        data=data.drop('subject',axis=1)
        for i in range(0,len(data['message'])):

            data['message'][i]= data['message'][i].lower()
            data['message'][i] = re.sub(r'[^\w\s+]','',data['message'][i])
            data['message'][i]=  data['message'][i].replace("_","")
    
        wordfrequency={}
        
        nltk.download('punkt')
        for text in data['message']:
            tokens=nltk.word_tokenize(text)
            for token in tokens:
                if token not in wordfrequency.keys():
                    wordfrequency[token]=1
                else:
                    wordfrequency[token]+=1
        
        most_frequency = heapq.nlargest(1000, wordfrequency, key=wordfrequency.get)

        
        sentence_vectors = []
        for sentence in data['message']:
            sentence_tokens = nltk.word_tokenize(sentence)
            sent_vec = []
            for token in most_frequency:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
            
        sentence_vectors = np.asarray(sentence_vectors)
        
        
        X=sentence_vectors
        y=data['label'].values
       

if(datasets[sys.argv[1]]==3):
        data=pd.read_csv(sys.argv[1])
   
       
        for i in range(0,len(data['text'])):
            data['text'][i]=data['text'][i].replace("Subject:","")
            data['text'][i]= data['text'][i].lower()
            data['text'][i] = re.sub(r'[^\w\s+]','',data['text'][i])
            data['text'][i]=  data['text'][i].replace("_","")
    
        wordfrequency={}
        
        nltk.download('punkt')
        for text in data['text']:
            tokens=nltk.word_tokenize(text)
            for token in tokens:
                if token not in wordfrequency.keys():
                    wordfrequency[token]=1
                else:
                    wordfrequency[token]+=1
        
        most_frequency = heapq.nlargest(1000, wordfrequency, key=wordfrequency.get)

        
        sentence_vectors = []
        for sentence in data['text']:
            sentence_tokens = nltk.word_tokenize(sentence)
            sent_vec = []
            for token in most_frequency:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
            
        sentence_vectors = np.asarray(sentence_vectors)
        
        
        X=sentence_vectors
        y=data['spam'].values
       

if(datasets[sys.argv[1]]==2):
   
   
        data=pd.read_csv(sys.argv[1], encoding = "latin-1")
        data=data.drop('Unnamed: 2',axis=1)
        data=data.drop('Unnamed: 3',axis=1)
        data=data.drop('Unnamed: 4',axis=1)
        encoder = LabelEncoder()
        data["v1"] = encoder.fit_transform(data["v1"])
        for i in range(0,len(data['v2'])):
            # data['v2'][i]=data['text'][i].replace("Subject:","")
            data['v2'][i]= data['v2'][i].lower()
            data['v2'][i] = re.sub(r'[^\w\s+]','',data['v2'][i])
            data['v2'][i]=  data['v2'][i].replace("_","")
    
        wordfrequency={}
        
        nltk.download('punkt')
        for text in data['v2']:
            tokens=nltk.word_tokenize(text)
            for token in tokens:
                if token not in wordfrequency.keys():
                    wordfrequency[token]=1
                else:
                    wordfrequency[token]+=1
        
        most_frequency = heapq.nlargest(1000, wordfrequency, key=wordfrequency.get)

        
        sentence_vectors = []
        for sentence in data['v2']:
            sentence_tokens = nltk.word_tokenize(sentence)
            sent_vec = []
            for token in most_frequency:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
            
        sentence_vectors = np.asarray(sentence_vectors)
        
        
        X=sentence_vectors
        y=data['v1'].values
       






if(datasets[sys.argv[1]]==4):
    
   
        
        data=pd.read_csv(sys.argv[1], sep=';')

        encoder = LabelEncoder()
        data["Column1"] = encoder.fit_transform(data["Column1"])
        for i in range(0,len(data['Column2'])):
            # data['v2'][i]=data['text'][i].replace("Subject:","")
            data['Column2'][i]= data['Column2'][i].lower()
            data['Column2'][i] = re.sub(r'[^\w\s+]','',data['Column2'][i])
            data['Column2'][i]=  data['Column2'][i].replace("_","")
    
        wordfrequency={}
        
        nltk.download('punkt')
        for text in data['Column2']:
            tokens=nltk.word_tokenize(text)
            for token in tokens:
                if token not in wordfrequency.keys():
                    wordfrequency[token]=1
                else:
                    wordfrequency[token]+=1
        
        most_frequency = heapq.nlargest(1000, wordfrequency, key=wordfrequency.get)

        
        sentence_vectors = []
        for sentence in data['Column2']:
            sentence_tokens = nltk.word_tokenize(sentence)
            sent_vec = []
            for token in most_frequency:
                if token in sentence_tokens:
                    sent_vec.append(1)
                else:
                    sent_vec.append(0)
            sentence_vectors.append(sent_vec)
            
        sentence_vectors = np.asarray(sentence_vectors)
        
        
        X=sentence_vectors
        y=data['Column1'].values
       




datasets_model={'MultinomialNB':1, 'SVC':2, 'Random Forest':3, 'K-Neighbors':4,'Deep Learning':5}



if(datasets_model[sys.argv[2]]==1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    
    
    classifier = MultinomialNB()
    classifier.fit(X_train , y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    print("Confussion Matrix for Multinomial Naive Bayes:")
    print(confusion_matrix(y_test, y_pred))

    
    print("Accuracy Score for Multinomial Naive Bayes:" )
    print(accuracy_score(y_test, y_pred))

    
    
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Mean for Multinomial Naive Bayes:")
    print(accuracies.mean())
    print("Standard Deviation for Multinomial Naive Bayes:" )
    print(accuracies.std())
    
    
    print("Classification report for Multinomial Naive Bayes")
    print(classification_report(y_test,y_pred,[0, 1]))







if(datasets_model[sys.argv[2]]==2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    
    
    classifier = SVC()
    classifier.fit(X_train , y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    print("Confussion Matrix for Support Vector Machine:")
    print(confusion_matrix(y_test, y_pred))

    
    print("Accuracy Score for Support Vector Machine:" )
    print(accuracy_score(y_test, y_pred))

    
    
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Mean for Multinomial Naive Bayes:")
    print(accuracies.mean())
    print("Standard Deviation for Support Vector Machine:" )
    print(accuracies.std())
    
    
    print("Classification report for Support Vector Machine")
    print(classification_report(y_test,y_pred,[0, 1]))





if(datasets_model[sys.argv[2]]==3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    
    
    classifier = RandomForestClassifier(n_estimators=200,criterion='gini', min_samples_split=3,verbose=0)
    classifier.fit(X_train , y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    print("Confussion Matrix for Random Forest:")
    print(confusion_matrix(y_test, y_pred))

    
    print("Accuracy Score for Random Forest:" )
    print(accuracy_score(y_test, y_pred))

    
    
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Mean for Random Forest:")
    print(accuracies.mean())
    print("Standard Deviation for Random Forest:" )
    print(accuracies.std())
    
    
    print("Classification report for Random Forest")
    print(classification_report(y_test,y_pred,[0, 1]))




if(datasets_model[sys.argv[2]]==4):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    
    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train , y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    print("Confussion Matrix for K-Neighbors:")
    print(confusion_matrix(y_test, y_pred))

    
    print("Accuracy Score for K-Neighbors:" )
    print(accuracy_score(y_test, y_pred))

    
    
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Mean for K-Neighbors:")
    print(accuracies.mean())
    print("Standard Deviation for K-Neighbors:" )
    print(accuracies.std())
    
    
    print("Classification report for K-Neighbors")
    print(classification_report(y_test,y_pred,[0, 1]))




# if(datasets_model[sys.argv[2]]==5):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    
    
#     model = Sequential()
#     model.add(Dense(500, activation='relu', input_dim=1000))
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])
#     history =model.fit(X_train, y_train, epochs=20,validation_data=(X_test,y_test))
#     # Predicting the Test set results
#     y_pred = model.predict(X_test)
#     scores = model.evaluate(X_train, y_train, verbose=1)
#     print('Accuracy on test data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 

    
    

  





