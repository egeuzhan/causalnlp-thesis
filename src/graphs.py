import plotly.graph_objects as go
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### Based on Plotly implementation in https://plotly.com/python/error-bars/

# For NER loss graphs
def ner_loss(model_path):
    rel_path = "loss.csv"
    path = os.path.join(model_path, rel_path)
    df = pd.read_csv(path)
    df2 = df.rename(columns={'train_loss': 'Training Loss', 'val_loss': 'Validation Loss'})
    df2[["Training Loss", "Validation Loss"]].ffill().plot(grid=True)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

# bar charts for RE
def re_graphs():

    # For CREST-1
    fig_re_crest1 = go.Figure()
    fig_re_crest1.add_trace(go.Bar(
        name = 'F1' ,
        x = ['CREST-1-Cause', 'CREST-1-Effect'],
        y = [5, 6],
        error_y = dict(type='data', array= [])

    ))
    fig_re_crest1.add_trace(go.Bar(
        name = 'Precision', 
        x = ['CREST-1-Cause', 'CREST-1-Effect'],
        y = [8, 29],
        error_y = dict(type='data', array= [])

    ))
    fig_re_crest1.add_trace(go.Bar(
        name = 'Recall', 
        x = ['CREST-1-Cause', 'CREST-1-Effect'],
        y = [4, 3],
        error_y = dict(type='data', array= [])

    ))
    fig_re_crest1.update_xaxes(title_font=dict(size=20))
    fig_re_crest1.update_layout(barmode='group',
                    showlegend=False,
                    font=dict(
                        family="Open Sans",
                        size=20,
                        color="Black"
        ))
    fig_re_crest1.show()
    fig_re_crest1.write_image("figner1.png")


    # For CREST-2
    fig_re_crest2 = go.Figure()
    fig_re_crest2.add_trace(go.Bar(
        name = 'F1' ,
        x = ['CREST-2-Cause', 'CREST-2-Effect'],
        y = [9, 6],
        error_y = dict(type='data', array= [])

    ))
    fig_re_crest2.add_trace(go.Bar(
        name = 'Precision', 
        x = ['CREST-2-Cause', 'CREST-2-Effect'],
        y = [7, 19],
        error_y = dict(type='data', array= [])

    ))
    fig_re_crest2.add_trace(go.Bar(
        name = 'Recall', 
        x = ['CREST-2-Cause', 'CREST-2-Effect'],
        y = [13, 4],
        error_y = dict(type='data', array= [])

    ))
    fig_re_crest2.update_xaxes(title_font=dict(size=20))
    fig_re_crest2.update_layout(barmode='group',
                    showlegend=False,
                    font=dict(
                        family="Open Sans",
                        size=20,
                        color="Black"
        ))
    fig_re_crest2.show()


    # For IE
    fig_re_ie = go.Figure()
    fig_re_ie.add_trace(go.Bar(
        name = 'F1' ,
        x = ['IE-Cause', 'IE-Effect'],
        y = [9, 6],
        error_y = dict(type='data', array= [])

    ))
    fig_re_ie.add_trace(go.Bar(
        name = 'Precision', 
        x = ['IE-Cause', 'IE-Effect'],
        y = [7, 19],
        error_y = dict(type='data', array= [])

    ))
    fig_re_ie.add_trace(go.Bar(
        name = 'Recall', 
        x = ['IE-Cause', 'IE-Effect'],
        y = [13, 4],
        error_y = dict(type='data', array= [])

    ))
    fig_re_ie.update_xaxes(title_font=dict(size=20))
    fig_re_ie.update_layout(barmode='group',
                    showlegend=False,
                    font=dict(
                        family="Open Sans",
                        size=20,
                        color="Black"
        ))
    fig_re_ie.show()