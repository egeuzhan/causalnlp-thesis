import json
import os
import re
import pandas as pd
import argparse

# creates a dataframe for the final results
def show_result(pred_path, source_path):

  # predictions
    df_result = pd.read_json(path_or_buf=pred_path)
    # original data for getting entity indexes
    df_original = pd.read_json(path_or_buf=source_path)
    df_print = pd.DataFrame(columns=['Title', 'Cause', 'Effect'])
    rel_count = 0
  # for each pair in the datasets
    for idx, elem in df_result.iterrows():
        for j, elem2 in df_original.iterrows():
            if elem['title'] == elem2['title']:
                title= elem['title']
                cause_id = elem['h_idx']
                effect_id = elem['t_idx']
                rel_id = elem['r']

                # get entity indexes from the original data
                cause = elem2['vertexSet'][cause_id]
                effect = elem2['vertexSet'][effect_id]

                cause = cause[0]
                effect = effect[0]
                df_print.loc[idx] = [title, cause, effect]
    # calculate total causal relations
    #for idx, elem in df_original.iterrows():
    #    labels = elem['labels']
    #    for label in labels:
    #        rel_count += 1
    with open('../predictions/final_predictions.json', 'w') as file:
        out = df_print.to_json(orient = 'records')
        file.write(out)

# implements the model argument for selecting the model
def main():
    parser = argparse.ArgumentParser()
  
    parser.add_argument("--model", default="atlop", type=str)
  
    args = parser.parse_args()

    if args.model == "atlop":
        show_result("../predictions/atlop/atlop_predictions.json", "../predictions/ner/test_ie_ent.json")
  
    if args.model == "docunet":
        show_result("../predictions/docunet/docunet_predictions.json", "../predictions/ner/test_ie_ent.json")


if __name__ == '__main__':
    main()