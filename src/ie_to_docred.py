import huggingface_hub
from datasets import get_dataset_split_names
import os
import re
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import json
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

# sentencizer and tokenizer
nlp =  spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
tokenizer2 = nlp.tokenizer
nlp.add_pipe('sentencizer')

# remove ! from sentence separation rules
suffixes = list(nlp.Defaults.suffixes)
suffixes.remove("\\!")
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

# get dataset (doccano)
myjsonl = "/impact_eval_annotated.jsonl"

json_obj = pd.read_json(path_or_buf=myjsonl, lines=True)

with open('/Users/ege/Downloads/ie_for_ner.json', 'w') as ds:
    for idx, line in json_obj.iterrows():
        if line['part'] == 0:
          continue
        # create new dataset for each section
        newDataset = pd.DataFrame(columns=['title', 'sents', 'vertexSet'])
        title = line['title']
        sentences = line['text']
        sentences2 = sentences.replace("\n", " ")
        
        # create sentence list
        doc = nlp(sentences2)
        sents = [sent.text.strip() for sent in doc.sents]
        sentlist= []
        for sentence in sents:
          s = [t.text for t in tokenizer2(sentence)]
          sentlist.append(s)

        # dictionaries for entity and relations
        entities = {}
        relations = {}
        vertexSet = []
        
        # get entities
        for entity in line['entities']:

            # get entity type
            if (entity['label'] == 'Effect'):
                entitytype= 'EFFECT'
            if (entity['label'] == 'Intervention'):
                entitytype = 'INT'

            if (entity['label'] == 'Cause'):
                entitytype = 'CAUSE'

            # find location of token from the character indexes
            # using the sentence list of tokens, 
            start = int(entity['start_offset'])
            end = int(entity['end_offset'])
            dist = end - start
            entityname = sentences[start:end]
            doc2= nlp(entityname)
            entitytokens = tokenizer2(entityname.strip())
            entitytokens = entitytokens[0:len(entityname)]
            occurrences = []
            # find the matching span inside the sentence, add to a list of occurrences
            for j, sent in enumerate(doc.sents):
              if entityname in sent.text:
                for i in range(len(sent)):
                  if (sent[i:i+len(entitytokens)].text == entitytokens.text):
                    occurrences.append((j, [i, i + len(entitytokens)]))
                
            for o in occurrences:
              sentid = o[0]
              pos = o[1]

              # for each occurence of the entity name, create a new entity dictionary
              newEntity = []
              entityDict = {'name': entityname, 'sent_id': sentid, 'pos': pos,'type': entitytype}
              if entitytype == 'INT' and any(e[0]['type'] == 'INT' for e in vertexSet):
                # put interventions as same entity
                for ent in vertexSet:
                  if ent[0]['type'] == 'INT':
                    ent.append(entityDict)
                    break
              else:
                newEntity.append(entityDict)
                vertexSet.append(newEntity)
              entities[entity['id']] = entityDict
            
        labels = []

        # get relations 
        for relation in line['relations']:
          if (relation['type'] == 'Cause\/Effect'):
            rel = 'ce'
            id = 'P9999'
          # get entity in the entity list through the index in the relation
          relfrom = relation['from_id']
          relto = relation['to_id']
          ent1 = entities[relfrom]
          ent2 = entities[relto]
          
          heads = []
          tails = []
          
          relid = 'P9999'

          # get sentence ids of entities as evidence
          evidence = []
          evidence.append(ent1['sent_id'])
          if not ent2['sent_id'] in evidence:
            evidence.append(ent2['sent_id'])
          
          # obtain each (head, tail) pair 
          for i, vertex in enumerate(vertexSet):
            for j, intervention in enumerate(vertex):
              if (ent1['name'] == vertex[j]['name']):
                heads.append(i)
                break

          for i, vertex in enumerate(vertexSet):
            if (ent2['name'] == vertex[0]['name']):
              tails.append(i)
              break
          # create dictionary for each relation, put into relation list
          for i, (head, tail)  in enumerate(zip(heads, tails)):
            rellabel = {'h': head, 't': tail, 'r': relid, 'evidence': evidence}
            labels.append(rellabel)

        # create final dictionary
        new_dict = {"title": title, "sents": sentlist, "vertexSet": vertexSet, "labels": labels}
        ds.write(json.dumps(new_dict) + '\n')
ds.close()
