from bi_lstm_crf.app import WordsTagger
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
import ast
import re
from itertools import groupby
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import json
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import shutil
import argparse

# Predict entities in test set and process into docred format
def ner_model(model_path, test_file_path):
    # get trained model 
    model = WordsTagger(model_dir=model_path, device = 'cpu')

    with open(test_file_path, 'r') as f:
        # predict each sentence one by one
        lines = f.readlines()
        y_true_list= []
        tagslist = []
        sentencelist= []
        titlelist = []
        for line in lines:
            title = line.split('\t')[0].strip()
            title = str(title)
            facts = line.split('\t')[2].strip()
            y_true = ast.literal_eval(facts)
            sent = line.split('\t')[1].strip()
            sentence = ast.literal_eval(sent)
            l1 = []
            l1.append(sentence)
            tags, sequences = model(l1)
            sentencelist.append(sentence)
            y_true_list.append(y_true)
            tagslist.append(tags[0])
            titlelist.append(title)

        #print(sentencelist)
        #print(y_true_list)
        #print(tagslist)

        # create list of all sentences/tags
        y_true_flat = [item for sublist in y_true_list for item in sublist]
        sentences_flat = [item for sublist in sentencelist for item in sublist]
        tags_flat = [item for sublist in tagslist for item in sublist]


        #Score table 
        #print(classification_report(y_true_flat, tags_flat, target_names=['C', 'E', 'O']))

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

        jsonlist = []

    # parse into docred format for the RE
    with open('../predictions/ner/test_ie_ent_pred.json', 'w') as f2:
        sent_id = 0
        for title, sent, tags in zip(titlelist, sentencelist, tagslist):
            newDataset = pd.DataFrame(columns=['title', 'sents', 'vertexSet'])
            # case for a section being added into a list where no other sections from same paper were added before
            if not any(elem['title'] == title for elem in jsonlist):
                sent_id = 0
                t = title
                ds = []
                sents = []
                sents.append(sent)
                # create lists for entities
                vertexSet = []
                causelist = []
                effectlist = []
                start_pos = 0
                end_pos = 0
                # look into previous and next tags to get a continuous entity,
                # start and end locations are also necessary
                prev_tag = 'O'
                for i, (token, tag) in enumerate(zip(sent, tags)):
                    if tag == 'C':
                        if prev_tag != 'C':
                            start_pos = i
                        if i == len(sent) - 1 or tags[i+1] != 'C':
                            end_pos = i
                            causelist.append((sent[start_pos:end_pos+1], (start_pos, end_pos + 1)))
                    if tag == 'E':
                        if prev_tag != 'E':
                            start_pos = i
                        if i == len(sentence) - 1 or tags[i+1] != 'E':
                            end_pos = i
                            effectlist.append((sent[start_pos:end_pos+1], (start_pos, end_pos + 1)))
                    prev_tag = tag

                # create entities from elements in cause and effect lists
                for elem1 in causelist:
                    ds_dict = {'name': elem1[0], 'pos': [elem1[1][0], elem1[1][1]], 'label': 'CAUSE', 'sent_id': sent_id}
                    ds.append(ds_dict)
                for elem2 in effectlist:
                    ds_dict = {'name': elem2[0], 'pos': [elem2[1][0], elem2[1][1]], 'label': 'EFFECT', 'sent_id': sent_id}
                    ds.append(ds_dict)
                if len(ds) > 0:
                    for elem in ds:
                        vertexSet.append([elem])

                # create final entity
                data = {'title': t, 'sents': sents, 'vertexSet': vertexSet}
                jsonlist.append(data)
            
            # case for a separate section of same paper already existing in list
            else:
                sent_id += 1
                for element in jsonlist:
                    if element['title'] == title:
                        element['sents'].append(sent)
                        # create lists for cause and effect entities
                        ds = []
                        causelist = []
                        effectlist = []
                        start_pos = 0
                        end_pos = 0
                        # look into previous and next tags to get a continuous entity,
                        # start and end locations are also necessary
                        prev_tag = 'O'
                        for i, (token, tag) in enumerate(zip(sent, tags)):
                            if tag == 'C':
                                if prev_tag != 'C':
                                    start_pos = i
                                if i == len(sent) - 1 or tags[i+1] != 'C':
                                    end_pos = i
                                    causelist.append((sent[start_pos:end_pos+1], (start_pos, end_pos + 1)))
                            if tag == 'E':
                                if prev_tag != 'E':
                                    start_pos = i
                                if i == len(sent) - 1 or tags[i+1] != 'E':
                                    end_pos = i
                                    effectlist.append((sent[start_pos:end_pos+1], (start_pos, end_pos + 1)))
                            prev_tag = tag

                # put each cause and effect entity in the vertexSet(entity list)
                for elem1 in causelist:
                    ds_dict = {'name': elem1[0], 'pos': [elem1[1][0], elem1[1][1]], 'label': 'CAUSE', 'sent_id': sent_id}
                    ds.append(ds_dict)
                for elem2 in effectlist:
                    ds_dict = {'name': elem2[0], 'pos': [elem2[1][0], elem2[1][1]], 'label': 'EFFECT', 'sent_id': sent_id}
                    ds.append(ds_dict)
                if len(ds) > 0:
                    for elem in ds: 
                        element['vertexSet'].append([elem])


        f2.write(json.dumps(jsonlist))
    f2.close()

    with open('../predictions/ner/test_ie_ent.json', 'w') as file:
        df = pd.read_json('../predictions/ner/test_ie_ent_pred.json')
        new_ds = pd.DataFrame(columns=['title', 'sents', 'vertexSet'])
        for idx, row in df.iterrows():
            cause = 0
            effect = 0
            for elem in row['vertexSet']:
                if elem[0]['label'] == 'CAUSE':
                    cause += 1
                if elem[0]['label'] == 'EFFECT':
                    effect += 1
            if cause > 1 and effect > 1:
                new_ds.loc[-1] = row
                new_ds.index = new_ds.index + 1
                new_ds = new_ds.sort_index()
        out = new_ds.to_json(orient='records')
        file.write(out)    

    # copy test set into each cv-set folder 
    src_path = "../predictions/ner/test_ie_ent.json"
    shutil.copy(src_path, "../data/preprocessed/re/crest_1/set1/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_1/set2/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_1/set3/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_1/set4/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_1/set5/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_2/set1/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_2/set2/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_2/set3/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_2/set4/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/crest_2/set5/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/ie/set1/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/ie/set2/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/ie/set3/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/ie/set4/test_ie_ent.json")
    shutil.copy(src_path, "../data/preprocessed/re/ie/set5/test_ie_ent.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="../data/results/ner/ie", type=str)
    parser.add_argument("--test_file_path", default="../data/test_ie.txt", type=str)

    args = parser.parse_args()
    if not args.model_path == "":
        ner_model(args.model_path, args.test_file_path)


if __name__ == '__main__':
    main()