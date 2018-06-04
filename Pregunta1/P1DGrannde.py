import os
import nltk
import re
import operator
import gensim 
import numpy as np
from stemming.porter2 import stem
from nltk.corpus import stopwords
#from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

dictionary = {}
dictionary_index = {}
exp_path_pos = '../aclImdb/train/pos/'
exp_path_neg = '../aclImdb/train/neg/'
exp_path_nt = '../aclImdb/train/unsup/'
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
stopW_without_punctuation = []

#documents = list()

for i in stopWords:
    stopW_without_punctuation.append(re.sub(r'[^\w\s]','',i))

for filename in os.listdir(exp_path_pos):
    with open(exp_path_pos+filename,'r') as f:
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    tempLine += temp_w + " "
                    if temp_w in dictionary:
                        dictionary[temp_w]=dictionary[temp_w] + 1
                    else:
                        dictionary[temp_w]=1
                    
            tempLine = tempLine.strip()
            #documents.append(tempLine.split())
            
for filename in os.listdir(exp_path_neg):
    with open(exp_path_neg+filename,'r') as f:
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    tempLine += temp_w + " "
                    if temp_w in dictionary:
                        dictionary[temp_w]=dictionary[temp_w] + 1
                    else:
                        dictionary[temp_w]=1

            tempLine = tempLine.strip()        
            #documents.append(tempLine.split())
            
for filename in os.listdir(exp_path_nt):
    with open(exp_path_nt+filename,'r') as f:
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    tempLine += temp_w + " "
                    if temp_w in dictionary:
                        dictionary[temp_w]=dictionary[temp_w] + 1
                    else:
                        dictionary[temp_w]=1

            tempLine = tempLine.strip()        
            #documents.append(tempLine.split())

print(len(dictionary))

sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
print(sorted_dictionary[0])
print(sorted_dictionary[1])
print(sorted_dictionary[2])
print(sorted_dictionary[3])
print(sorted_dictionary[4])

print(sorted_dictionary[-1])
print(sorted_dictionary[-2])
print(sorted_dictionary[-3])
print(sorted_dictionary[-4])
print(sorted_dictionary[-5])

dictKeys = list(dictionary.keys())
for i in range(len(dictKeys)):
    dictionary_index[dictKeys[i]]=i

bow_cods = []
#w2v_cods = []
labels = []
#model = gensim.models.Word2Vec (documents, size=100, window=10, min_count=2, workers=10)
#model.train(documents,total_examples=len(documents),epochs=10)

for filename in os.listdir(exp_path_pos):
    with open(exp_path_pos+filename,'r') as f:
        bow_temp = [0]*len(dictionary)
        #w2v_temp = np.empty((0,100))
        count = 0
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    bow_temp[dictionary_index[temp_w]]+=1
                    #if (temp_w in model.wv.vocab):
                    #    w2v_resize = np.array(model.wv[temp_w])
                    #    w2v_resize = np.resize(w2v_resize,(1,100))
                    #    w2v_temp = np.append(w2v_temp,w2v_resize, axis=0)
        #w2v_temp = np.mean(w2v_temp, axis=0)
        bow_cods.append(bow_temp)
        labels.append("pos")
        #w2v_cods.append([w2v_temp, "pos"])
        
for filename in os.listdir(exp_path_neg):
    with open(exp_path_neg+filename,'r') as f:
        bow_temp = [0]*len(dictionary)
        #w2v_temp = np.empty((0,100))
        count = 0
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    bow_temp[dictionary_index[temp_w]]+=1
                    #if (temp_w in model.wv.vocab):
                    #    w2v_resize = np.array(model.wv[temp_w])
                    #    w2v_resize = np.resize(w2v_resize,(1,100))
                    #    w2v_temp = np.append(w2v_temp,w2v_resize, axis=0)
        #w2v_temp = np.mean(w2v_temp, axis=0)
        bow_cods.append(bow_temp)
        labels.append("neg")
        #w2v_cods.append([w2v_temp, "neg"])
        


lab_fold_train_bow = []
#lab_fold_train_w2v = []
lab_fold_test_bow = []
#lab_fold_test_w2v = []

con_fold_train_bow = []
#con_fold_train_w2v = []
con_fold_test_bow = []
#con_fold_test_w2v = []

#TRAIN
#for i in bow_cods:
#    con_fold_train_bow.append(i[0])
#    lab_fold_train_bow.append(i[1])
#for i in w2v_cods:
#    con_fold_train_w2v.append(i[0])
#    lab_fold_train_w2v.append(i[1])
#TEST    
#for i in bow_cods_test:
#    con_fold_test_bow.append(i[0])
#    lab_fold_test_bow.append(i[1])
#for i in w2v_cods_test:
#    con_fold_test_w2v.append(i[0])
#    lab_fold_test_w2v.append(i[1])
    
svm_bow = LinearSVC() 
#svm_w2v = LinearSVC()

svm_bow.fit(bow_cods, labels) 
#svm_w2v.fit(con_fold_train_w2v, lab_fold_train_w2v) 

print("Training Set Accuracy BOW:", svm_bow.score(bow_cods, labels))
print("Deleating training data")
del(bow_cods)
del(labels)
print("Training data deleted")


#TEST
exp_path_pos_test = '../aclImdb/test/pos/'
exp_path_neg_test = '../aclImdb/test/neg/'
bow_cods_test = []
labels_test = []
#w2v_cods_test = []
for filename in os.listdir(exp_path_pos_test):
    with open(exp_path_pos_test+filename,'r') as f:
        bow_temp = [0]*len(dictionary)
        #w2v_temp = np.empty((0,100))
        count = 0
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    if (temp_w not in dictionary_index):
                        continue
                    bow_temp[dictionary_index[temp_w]]+=1
                    #if (temp_w in model.wv.vocab):
                    #    w2v_resize = np.array(model.wv[temp_w])
                    #    w2v_resize = np.resize(w2v_resize,(1,100))
                    #    w2v_temp = np.append(w2v_temp,w2v_resize, axis=0)
        #w2v_temp = np.mean(w2v_temp, axis=0)
        bow_cods_test.append(bow_temp)
        labels_test.append("pos")
        #w2v_cods_test.append([w2v_temp, "pos"])
        
for filename in os.listdir(exp_path_neg_test):
    with open(exp_path_neg_test+filename,'r') as f:
        bow_temp = [0]*len(dictionary)
        #w2v_temp = np.empty((0,100))
        count = 0
        for line in f:
            #delete Punctuation
            line = re.sub(r'[^\w\s]','',line)
            tempLine = ""
            for word in line.split():
                word = word.lower()
                #Stop Words
                if word not in stopW_without_punctuation:
                    temp_w = stem(word)
                    if (temp_w not in dictionary_index):
                        continue
                    bow_temp[dictionary_index[temp_w]]+=1
                    #if (temp_w in model.wv.vocab):
                    #    w2v_resize = np.array(model.wv[temp_w])
                    #    w2v_resize = np.resize(w2v_resize,(1,100))
                    #    w2v_temp = np.append(w2v_temp,w2v_resize, axis=0)
        #w2v_temp = np.mean(w2v_temp, axis=0)
        bow_cods_test.append(bow_temp)
        labels_test.append("neg")
        #w2v_cods_test.append([w2v_temp, "neg"])

print("Testing Set Accuracy BOW:", svm_bow.score(bow_cods_test, labels_test))
#print("Training Set Accuracy W2V:", svm_w2v.score(con_fold_train_w2v, lab_fold_train_w2v))
#print("Testing Set Accuracy W2V:", svm_w2v.score(con_fold_test_w2v, lab_fold_test_w2v))
print("")
print ('\nConfussion matrix BOW:\n',confusion_matrix(labels_test, svm_bow.predict(bow_cods_test)))
#print ('\nConfussion matrix W2V:\n',confusion_matrix(lab_fold_test_w2v, svm_w2v.predict(con_fold_test_w2v)))
print("")


