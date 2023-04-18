import json
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import random


# splits dataset in 70/15/15
def re_split_train_test_val(crest_path):
    file = json.load(open(crest_path, "r"))

    train, test = train_test_split(file, test_size =0.3)

    val, test = train_test_split(test, test_size = 0.5)

    with open("train_crest.json", "w") as f:
        f.write(json.dumps(train))
        print(len(train))

    with open("test_crest.json", "w") as f:
        f.write(json.dumps(test))
        print(len(test))

    with open("val_crest.json", "w") as f:
        f.write(json.dumps(val))
        print(len(val))



# creates shuffled 5-fold cross validation sets for crest
def re_crest_split_train_val(dataset_path):

    df = pd.read_json(path_or_buf=dataset_path)

    # shuffle the dataset and put into dataframes
    shuffled = df.sample(frac=1)
    splits = np.array_split(shuffled, 5)
    df1 = splits[0]
    df2 = splits[1]
    df3 = splits[2]
    df4 = splits[3]
    df5 = splits[4]

    # merge four different dataframes into one, for each fold
    #1:
    train1list = [df1, df2, df3, df4]
    test1 = df5
    train1 = pd.concat(train1list)
    train1json = json.dumps(train1.to_dict(orient='records'))  
    test1json = json.dumps(test1.to_dict(orient='records'))              
    with open('/set1/train_annotated.json', 'w') as f:
        f.write(train1json)
    with open('/set1/test.json', 'w') as f:
        f.write(test1json)

    #2:
    train2list = [df1, df2, df3, df5]
    test2 = df4
    train2 = pd.concat(train2list)
    train2json = json.dumps(train2.to_dict(orient='records'))  
    test2json = json.dumps(test2.to_dict(orient='records'))              
    with open('/set2/train_annotated.json', 'w') as f:
        f.write(train2json)
    with open('/set2/test.json', 'w') as f:
        f.write(test2json)

    #3:
    train3list = [df1, df2, df4, df5]
    test3 = df3
    train3 = pd.concat(train3list)
    train3json = json.dumps(train3.to_dict(orient='records'))  
    test3json = json.dumps(test3.to_dict(orient='records'))              
    with open('/set3/train_annotated.json', 'w') as f:
        f.write(train3json)
    with open('/set3/test.json', 'w') as f:
        f.write(test3json)

    #4:
    train4list = [df1, df3, df4, df5]
    test4 = df2
    train4 = pd.concat(train4list)
    train4json = json.dumps(train4.to_dict(orient='records'))  
    test4json = json.dumps(test4.to_dict(orient='records'))              
    with open('/set4/train_annotated.json', 'w') as f:
        f.write(train4json)
    with open('/set4/test.json', 'w') as f:
        f.write(test4json)

    #5:
    train5list = [df2, df3, df4, df5]
    test5 = df1
    train5 = pd.concat(train5list)
    train5json = json.dumps(train5.to_dict(orient='records'))  
    test5json = json.dumps(test5.to_dict(orient='records'))              
    with open('/set5/train_annotated.json', 'w') as f:
        f.write(train5json)
    with open('/set5/test.json', 'w') as f:
        f.write(test5json)

# creates shuffled 5-fold cross-validation sets for IE
def re_ie_split_train_test(ie_path):

    df = pd.read_json(path_or_buf=ie_path)

    # classify every section belonging to same paper
    docs = []
    for idx, elem in df.iterrows():
        i = 0
        if len(docs) == 0:
            doc = []
            doc.append(elem)
            docs.append(doc)
            continue
        for doc in docs:
            title = elem['title']
            if doc[0]['title'][:-1] == title[:-1]:
                doc.append(elem)
                i = 1
                break
        if i == 1:
            continue
        doc = []
        doc.append(elem)
        docs.append(doc)

    # shuffle and split, test_ie is created through the test variable
    random.shuffle(docs)

    train, test = train_test_split(docs, test_size=0.15)

    #val, test = train_test_split(test1, test_size=0.5)

    # create dataframes of all sections of each paper
    set1 = train[0:11]
    set2 = train[11:21]
    set3 = train[21:31]
    set4 = train[31:41]
    set5 = train[41:51]

    list1 = [item for sublist in set1 for item in sublist]
    list2 = [item for sublist in set2 for item in sublist]
    list3 = [item for sublist in set3 for item in sublist]
    list4 = [item for sublist in set4 for item in sublist]
    list5 = [item for sublist in set5 for item in sublist]
    listtest = [item for sublist in test for item in sublist]

    df1 = pd.DataFrame(list1)
    df2 = pd.DataFrame(list2)
    df3 = pd.DataFrame(list3)
    df4 = pd.DataFrame(list4)
    df5 = pd.DataFrame(list5)
    dftest = pd.DataFrame(listtest)

    testjson = json.dumps(dftest.to_dict(orient='records'))

    # merge four different dataframes into one, for each fold
    #1:
    train1list = [df1, df2, df3, df4]
    print(train1list)
    val1 = df5

    train1 = pd.concat(train1list)
    train1json = json.dumps(train1.to_dict(orient='records'))  
    val1json = json.dumps(val1.to_dict(orient='records'))              
    with open('/set1/train_annotated.json', 'w') as f:
        f.write(train1json)
    with open('/set1/val.json', 'w') as f:
        f.write(val1json)
    with open('/set1/test.json', 'w') as f:
        f.write(testjson)

    #2:
    train2list = [df1, df2, df3, df5]
    val2 = df4
    train2 = pd.concat(train2list)
    train2json = json.dumps(train2.to_dict(orient='records'))  
    val2json = json.dumps(val2.to_dict(orient='records'))              
    with open('/set2/train_annotated.json', 'w') as f:
        f.write(train2json)
    with open('/set2/val.json', 'w') as f:
        f.write(val2json)
    with open('/set2/test.json', 'w') as f:
        f.write(testjson)

    #3:
    train3list = [df1, df2, df4, df5]
    val3 = df3
    train3 = pd.concat(train3list)
    train3json = json.dumps(train3.to_dict(orient='records'))  
    val3json = json.dumps(val3.to_dict(orient='records'))              
    with open('/set3/train_annotated.json', 'w') as f:
        f.write(train3json)
    with open('/set3/val.json', 'w') as f:
        f.write(val3json)
    with open('/set3/test.json', 'w') as f:
        f.write(testjson)

    #4:
    train4list = [df1, df3, df4, df5]
    val4 = df2
    train4 = pd.concat(train4list)
    train4json = json.dumps(train4.to_dict(orient='records'))  
    val4json = json.dumps(val4.to_dict(orient='records'))              
    with open('/set4/train_annotated.json', 'w') as f:
        f.write(train4json)
    with open('/set4/val.json', 'w') as f:
        f.write(val4json)
    with open('/set4/test.json', 'w') as f:
        f.write(testjson)

    #5:
    train5list = [df2, df3, df4, df5]
    val5 = df1
    train5 = pd.concat(train5list)
    train5json = json.dumps(train5.to_dict(orient='records'))  
    val5json = json.dumps(val5.to_dict(orient='records'))              
    with open('/set5/train_annotated.json', 'w') as f:
        f.write(train5json)
    with open('/set5/dev.json', 'w') as f:
        f.write(val5json)
    with open('/set5/test.json', 'w') as f:
        f.write(testjson)



# fix the cause and effect labelings,
def fix_relations(ds_path):
    df = pd.read_json(path_or_buf=ds_path)

    lines= []
    for idx, elem in df.iterrows():
        rels = elem['labels']
        for rel in rels:
            h = rel['h']
            t = rel['t']
            
            
            elem1 = elem['vertexSet'][h]
            elem2 = elem['vertexSet'][t]
            elem1[0]['type'] = 'CAUSE'
            elem2[0]['type'] = 'EFFECT'

    out = df.to_json('/ds_fix.json', orient='records', lines=False)
    
# merge train and validation sets for the NER model
def ner_merge_train_val():

    df1 = pd.read_json(path_or_buf="/set1/train_annotated.json")
    df2 = pd.read_json(path_or_buf="/set1/val.json")

    train1list = [df1, df2]
    train1 = pd.concat(train1list)
    train1json = json.dumps(train1.to_dict(orient='records'))             
    with open('/set1/train_annotated_merged.json', 'w') as f:
        f.write(train1json)






    