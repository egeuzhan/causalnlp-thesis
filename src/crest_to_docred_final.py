import os
import re
import math
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import ast
import spacy
from spacy.lang.en import English
import json



# reads crest from original excel file
def get_ds(path):
    dataset = pd.read_excel(path)
    print(dataset.head())
    return dataset

# parses the document into docred format
def crest_to_docred(dataset):
    # tokenizer and sentencizer
    nlp = spacy.load('en_core_web_sm')
    tokenizer = nlp.tokenizer
    nlp.add_pipe('sentencizer')

    with open('dataset.jsonl', 'w') as docred:
        for idx, row in dataset.iterrows():
            title = row['original_id']
            sent = row['context'].replace('\n', ' ')
            sentlist = []
            # create list of sentences with tokens
            s = [t.text for t in tokenizer(sent)]
            sentlist.append(s)

            # parse string into list of spans
            span1 = ast.literal_eval(row['span1'])
            span2 = ast.literal_eval(row['span2'])
            
            # ignore empty strings or any span with multiple elements
            if len(span1) == 0:
                span1 = ['']
            if len(span2) == 0:
                span2 = ['']
            if '' in span1 or '' in span2 or row['span1'] == '' or row['span2'] == '' or row['span1'] == ' ' or row['span2'] == ' ' or row['span1'] == '[]' or row['span2'] == '[]' or row['span1'] == '['']' or row['span2'] == '['']' or row['span1'] == '[""]' or row['span2'] == '[""]' or row['span1'] == '["]' or row['span2'] == '["]':
              continue
            if len(span1) > 1 or len(span2) > 1:
              continue
            
            line = row['idx'].replace('\n', '')

            # obtain span locations in text
            line1 = line.split('span2')[0].split('span1')[1].strip()
            span1loc = (line1.split(':')[0], line1.split(':')[1])

            line2 = line.split('span2')[1].split('signal')[0].strip()
            span2loc = (line2.split(':')[0], line2.split(':')[1])

            # rows for label and direction
            label = row['label']
            direction = row['direction']

            title = str(idx)
            name1 = span1[0]
            name2 = span2[0]

            vertexSet = []
            poslist = []
            sent_id1 = None
            sent_id2 = None
            
            # get token indexes from character indexes
            start1 = int(span1loc[0])
            end1 = int(span1loc[1])
            start2 = int(span2loc[0])
            end2 = int(span2loc[1])

            fulltext = sent
            docu = nlp(fulltext)
            first = docu.char_span(start1, end1)
            second = docu.char_span(start2, end2)

            # skip the cases not parseable
            if (first == None or second ==None):
              continue

            # get token locations in sentences (for crest_2)
            # we need the sentence id and the location 
            # of tokens in that sentence
            startfirst = first[0].i
            endfirst = startfirst + len(first)
            sent_id1 = 0
            startsecond = second[0].i
            endsecond = startsecond + len(second)
            sent_id2 = 0
            pos1 = (startfirst, endfirst)
            loc1 = (sent_id1, pos1)
            pos2 = (startsecond, endsecond)
            loc2 = (sent_id2, pos2)

            ### for crest_1:
            #####
            # for j, sent in enumerate(doc.sents):
            #  sent_start = sent.start
            #  if first[0].i >= sent_start and first[0].i < sent_start + len(sent) - 1:
            #    start = first[0].i - sent_start
            #    end = start + len(first)
            #    sent_id1 = j
            #    pos1 = (start, end )
            #   loc1 = (j, pos1)

            #for j, sent in enumerate(doc.sents):
             # sent_start = sent.start
            #  if second[0].i >= sent_start and second[0].i < sent_start + len(sent) - 1:
            #    start = second[0].i - sent_start
             #   end = start + len(second)
             #   sent_id2 = j
             #   pos2 = (start, end )
            #    loc2 = (j, pos2)
             #   break
            #if sent_id1 == None or sent_id2 == None:
            #  continue     
            ######       
            
            # add each pair into the list of positions
            poslist.append(loc1)
            poslist.append(loc2)
            enttype = 'P9999'

            # create dictionary for entity
            entitydict1 = {'name': name1, 'sent_id': sent_id1, 'pos': [poslist[0][1][0], poslist[0][1][1]], 'type': enttype}
            entitydict2 = {'name': name2, 'sent_id': sent_id2, 'pos': [poslist[1][1][0], poslist[1][1][1]], 'type': enttype}

            # mention list contains all mentions of each 
            # entity, mainly used for interventions
            # vertexSet is made of mention lists of each entity
            mentionlist = []
            mentionlist.append(entitydict1)
            vertexSet.append(mentionlist)
            mentionlist = []
            mentionlist.append(entitydict2)
            vertexSet.append(mentionlist)

            labels = []
            evidence = []
            
            # relation is labeled as P9999
            r = 'P9999'

            # add evidences (sentence of entity)
            evidence.append(sent_id1)
            if sent_id2 not in evidence:
                evidence.append(sent_id2)
                
            # get direction of relation
            if direction == 0:
              h = 0
              t = 1
            elif direction == 1:
              h = 1
              t = 0
            else:
              h = 0
              t = 1

            # for non causal relations
            if label == 0:
                r = 'Na'

            # create dictionary for relation
            labeldict = {'h': h,'t': t,'r': r,'evidence': evidence}
            labels.append(labeldict)

            # final dictionary for the section(document)   
            newDict = {'title': title, 'sents': sentlist, 'vertexSet': vertexSet, 'labels': labels}

            endcount = endcount + 1
            docred.write(json.dumps(newDict) + '\n')
        
    # fixes label naming
       
    lines= []
    df1 = pd.read_json(path_or_buf="dataset.jsonl", lines = True)

    # annotate Cause or Effect for each entity
    for idx, elem in df1.iterrows():
        rels = elem['labels']
        for rel in rels:
            h = rel['h']
            t = rel['t']
            
            elem1 = elem['vertexSet'][h]
            elem2 = elem['vertexSet'][t]
            elem1[0]['type'] = 'CAUSE'
            elem2[0]['type'] = 'EFFECT'

    out = df1.to_json('dataset_fix.jsonl', orient='records', lines=True)

    # parse jsonl into json
    with open("dataset_fix.jsonl", 'r') as infile:
        with open('dataset_fix.json', 'w') as ds:
            #for idx, line in jsonObj.iterrows():
                lines = [json.loads(line) for line in infile.readlines()]
                json.dump(lines, ds)


crest_to_docred(get_ds("/crest_v2_fix.xlsx"))





        