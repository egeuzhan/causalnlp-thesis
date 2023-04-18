import pandas as pd
import os 
import re
import os
import re
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import json


# Parses a jsonl file with special 
# keys from the path data for doccano
def create_jsonl_fix(path):
    # store each separate section
    files = [f for f in listdir(path) if isfile(join(path, f)) & f.endswith(".txt")]
    ds = []

    filedict = {}

    # put section number/id in the title
    for idx, filename in enumerate(files):
        sep = '[0-6]'
        head = re.split(sep, filename, 1)[0]
        if head in filedict:
            ds[filedict[head]].append(filename)
        else:
            ds.append([filename])
            filedict[head] = ds.index([filename])
        
    for list in ds:
        list.sort()

    count = 0
    # create special keys for each section 
    with open("doccano_ds.txt", 'w') as newFile:
        for text in ds:
            for section in text:
                keys = ("id", "text", "entities", "relations", "title", "part", "Comments")
                jsonl = dict.fromkeys(keys)
                jsonl['id'] = count
                count += 1
                with open(path + '/' + section, 'r') as text:
                    textelems = text.readlines()
                    jsonl['text'] = ''.join(textelems)
                text.close()
                jsonl['entities'] = []
                jsonl['relations'] = []
                jsonl['title'] = re.split('[0-6]', section, 1)[0]
                jsonl['part'] = os.path.splitext(section)[0][-1]
                jsonl['Comments'] = []

                newFile.write(json.dumps(jsonl))
                newFile.write("\n")
    newFile.close() 


# removes longer sequences from the data path
def remove_long_seq(dataset_path):

    df = pd.read_json(path_or_buf=dataset_path)
    
    count_strings = lambda x: len(x)

    filtered_rows = []
    # filter rows longer than 1024 (max accepted by RE models)
    for index, row in df.iterrows():
        if len(row['sents'][0]) <= 1024:
            filtered_rows.append(row)
    df2 = pd.DataFrame(filtered_rows)
    
    # save all in new json file 
    test1json = json.dumps(df2.to_dict(orient='records'))              
    with open('/ds_short.json', 'w') as f:
        #for word in vocab:
        f.write(test1json)


# changes all intervention labels to cause
def int_to_cause(dataset_path):
    ds = pd.read_json(path_or_buf=dataset_path)
    
    for idx, doc in ds.iterrows():
        vset = doc['vertexSet']
        for mention in vset:
            for elem in mention:
                print(elem)
                if elem['type'] == 'INT':
                    elem['type'] = 'CAUSE'

    # save on new file
    out = ds.to_json('/ds_no_int.json', orient='records', lines=False)



# parses the given dataset into an 
# acceptable format for the NER model
def create_ner_input(dataset_path):
    df = pd.read_json(path_or_buf=dataset_path)

    sents = []
    tags = []
    titles = []

    for idx, elem in df.iterrows():
        title = elem['title']
        sentences = elem['sents']
        vertexSet = elem['vertexSet']
        newVertexSet = []
        # create new set of entities without lists, 
        # intervention entities are separated
        for idx, vertex in enumerate(vertexSet):
            if len(elem['vertexSet'][idx]) > 1:
                mentionlist = elem['vertexSet'].pop(idx)
                vertexSet.append(mentionlist)
                break
            newVertexSet.append(vertex)

       # tag each token with C or E
       # go through each sentence and each token, check
       # if token is in the sentence, then add tags
        for j, sent in enumerate(sentences):
            senttag = []
            for idx, token in enumerate(sent):
                for ent in newVertexSet:
                    if (idx < ent[0]['pos'][1]) and (idx >= ent[0]['pos'][0]) and (token in ent[0]['name']) and ent[0]['sent_id'] == j:
                        entity = newVertexSet.index(ent)
                        break
                    else: 
                        entity = ''
                
                if entity == '':
                    senttag.append('O')
                else: 
                    if newVertexSet[entity][0]['type'] == 'CAUSE':
                        senttag.append('C')
                    elif newVertexSet[entity][0]['type'] == 'EFFECT':
                        senttag.append('E')
                    elif newVertexSet[entity][0]['type'] == 'INT':
                        senttag.append('C')
                    
            # add each to a list
            tags.append(senttag)
            sents.append((sent, elem['title'], j))
            titles.append(title)
                    
            
    # puts each list of tags and tokens in a txt file (accepted format)
    with open('/dataset.txt', 'w') as f:
        for title, sent, tag in zip(titles, sents, tags):
            f.write(json.dumps(title) + "\t" + json.dumps(sent[0]) + "\t" + json.dumps(tag) + '\n')
    
    



# for creating the vocabulary file for the NER model
def create_ner_vocab(dataset_path):
    df = pd.read_json(path_or_buf=dataset_path)

    sents = []
    vocab = []

    # add each token in a list
    for idx, elem in df.iterrows():
        sentences = elem['sents']
        for j, sent in enumerate(sentences):
            for idx, token in enumerate(sent):
                if token not in vocab:
                    vocab.append(token)

    # create new txt file for vocabulary
    json_object = json.dumps(vocab)                
    with open('/vocab.json', 'w') as f:
        f.write(json_object)

# for separating the document into sections 
def separate_paragraphs(impact_eval_path, result_path):
    dataset = separate_sections(impact_eval_path)
    create_text_files(result_path, dataset)


# separates each section and returns a dataset of all sections
def separate_sections(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) & f.endswith('.txt')]
    files = [path + '/' + s for s in onlyfiles]
    data = []
    # check for 2+ whitespaces 
    # add into a new dataset
    for file in files:
        f = []
        with open(file, 'r') as file_data:
            n = False
            p = []
            for line in file_data:
                if line.isspace():
                    if n == True:
                        n = False
                        f.append(p)
                        p = []
                        continue
                    else: 
                        n = True
                        continue
                else: 
                    n = False
                p.append(line)
            f.append(p)
        file_data.close()
        data.append(f)
    return data

# creates files for each section in the input dataset
def create_text_files(out_path, data):
    for file in data:
        for idx, paragraph in enumerate(file):
            with open(out_path + '/' + file[0][0][:-1] + str(idx) + '.txt', 'w') as newFile:
                newFile.writelines(paragraph)
                #for elem in paragraph:
                #    newFile.write(elem)
                #    newFile.write('\n')
            newFile.close()



