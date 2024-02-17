from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import plotly
import json
import utils as ut
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
from datagenerator import DataGenerator
from collections import defaultdict
from joblib import dump, load
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import time 

app = Flask(__name__)



@app.route('/')
def index():

    # Update data
    # Main plot
    main_plot = make_big_plot()
    main_plot_json = json.dumps(main_plot, cls=plotly.utils.PlotlyJSONEncoder)

    model_perf = model_performance_plots(df)
    model_perf_json = json.dumps(model_perf, cls=plotly.utils.PlotlyJSONEncoder)

    small_plots = make_small_plots(df)
    small_plots_json = json.dumps(small_plots, cls=plotly.utils.PlotlyJSONEncoder)

    # Render template
    return render_template('index.html', 
                           main_plot_json=main_plot_json, 
                           model_perf_json=model_perf_json,
                           small_plots_json=small_plots_json)


@app.route('/get-new-data')
def get_new_data():
    # Generate new data
    global df, df_new_model_dev, rmse_new, model_version
    df = update_df(df)
    preds, new_model_flag, df_new_model_dev_, rmse_new = scoring(df)
    df['preds'] = preds
    if new_model_flag:
        model_version += 1
        df_new_model_dev = df_new_model_dev_
    plot_data = get_plot_data(df, new_model_flag, df_new_model_dev, rmse_new)

    return jsonify(plot_data)

def update_df(df=None):
    MAX_RECORDS = 30

    # Append new records
    n_obs_to_generate  = np.random.randint(100, 500)
    if df is None:
        df = dg.generate_data(n_obs_to_generate)
    else:
        df = pd.concat([df, dg.generate_data(n_obs_to_generate)], axis=0, ignore_index=True)

    # Remove oldest record if data gets too big
    if df['timestamp'].unique().shape[0] > MAX_RECORDS:
        df = df[df['timestamp'] != df['timestamp'].min()]

    return df

def scoring(df):
    model = load('model.joblib')
    x = df[['x0', 'x1', 'x2', 'x3']].iloc[-1000:]
    y = df['y'].iloc[-1000:]
    preds = model.predict(x)
    preds_all = model.predict(df[['x0', 'x1', 'x2', 'x3']])

    rmse_current = np.power(mean_squared_error(y, preds), 0.5)
    model.fit(x, y)
    preds_new = model.predict(x)
    rmse_new = np.power(mean_squared_error(y, preds_new), 0.5)
    print(f'{rmse_new} vs {rmse_current}')
    new_model_flag = False
    if rmse_new < rmse_current * 0.9:
        new_model_flag = True
        dump(model, 'model.joblib')
        print(f'Saving new model: rmse new:{rmse_new} vs rmse old:{rmse_current}')

    return preds_all, new_model_flag, x, rmse_new

def evaulate(y_true, y_pred):
    metrics = {}
    metrics['rmse'] = np.power(mean_squared_error(y_true, y_pred), 0.5)
    metrics['precision'] = r2_score(y_true, y_pred)
    
    return metrics

def get_plot_data(df, new_model_flag, df_new_model_dev, rmse_new):

    plot_data = defaultdict(dict)
    group_df = df.groupby('timestamp').agg({'x0':'count', 'y': 'mean', 'preds':'mean'}).reset_index()
    x = group_df['timestamp'].values.tolist()
    y1 = group_df['x0'].values.tolist()
    y2 = group_df['y'].values.tolist()
    y3 = group_df['preds'].values.tolist()

    plot_data['main'] = {
        'x': [x, x, x],
        'y': [y1, y2, y3]
    }

    rmse = df.groupby('timestamp').apply(lambda x: np.power(mean_squared_error(x['y'], x['preds']), 0.5), include_groups=False).reset_index(drop=True)
    r2 = df.groupby('timestamp').apply(lambda x: r2_score(x['y'], x['preds']), include_groups=False).reset_index(drop=True)
    x = df['timestamp'].unique().tolist()

    plot_data['model_perf'] = {
        'x':[x, x],
        'y':[rmse.values.tolist(), r2.values.tolist()]
    }

    p1_1 = df_new_model_dev['x0'].value_counts().sort_index()
    p1_2 = df['x0'].iloc[-1000:].value_counts().sort_index()

    p2_1 = df_new_model_dev['x1'].value_counts().sort_index()
    p2_2 = df['x1'].iloc[-1000:].value_counts().sort_index()

    p3_1 = df_new_model_dev['x2'].value_counts().sort_index()
    p3_2 = df['x2'].iloc[-1000:].value_counts().sort_index()

    p4_1 = df_new_model_dev['x3'].value_counts().sort_index()
    p4_2 = df['x3'].iloc[-1000:].value_counts().sort_index()

    plot_data['subplots'] = {
        'x':[p1_1.index.tolist(), 
             p1_2.index.tolist(),
             p2_1.index.tolist(),
             p2_2.index.tolist(), 
             p3_1.index.tolist(), 
             p3_2.index.tolist(),
             p4_1.index.tolist(),
             p4_2.index.tolist()],
        'y':[p1_1.values.tolist(), 
             p1_2.values.tolist(),
             p2_1.values.tolist(),
             p2_2.values.tolist(), 
             p3_1.values.tolist(), 
             p3_2.values.tolist(),
             p4_1.values.tolist(),
             p4_2.values.tolist()],
    }

    plot_data['model_update_flag'] = 0
    if new_model_flag:
        plot_data['model_update_flag'] = 1
        plot_data['model_update_values'] = {}
        plot_data['model_update_values']['version'] = model_version
        plot_data['model_update_values']['update_time'] = time.now()
        plot_data['model_update_values']['rmse'] = rmse_new
    
    return plot_data

def make_big_plot():
    """
    Create plotly price charts.
    """
    fig = go.Figure()
    x = [x for x in range(10)]
    y = [y for y in range(10)]
    fig.add_trace(go.Bar(x=x, y=y, name='Volume', opacity=0.7))
    fig.add_trace(go.Scatter(x=x, y=y, name='Trans Amount', yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=y, name='Preds', yaxis="y2", line={'dash':'dash'}))
    fig.update_layout(width=1200, 
                        height=500, 
                        yaxis_range=[0, 16000], 
                        title_text="<b>Transaction Volumes</b>", 
                        paper_bgcolor='#222222', 
                        font={'color':'#dadada'}, 
                        template='plotly_dark',
                        title={'x':0.5, 
                                'font': {'color':'#BB86FC'}},
                        yaxis={'title': 'Volume'},
                        yaxis2={
                                'title':'Trans Amount',
                                'overlaying':'y',  # This places y2 on top of y
                                'side':'right'  # This places y2 on the right side
                        },
                        legend={
                            'orientation':'h',  # Horizontal orientation
                            'yanchor':'bottom',
                            'y':-0.2,  # Adjust this value as needed to position the legend below the subplots
                            'xanchor':'center',
                            'x':0.5
                        }
    )
    return fig

def model_performance_plots(df):
    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("RMSE", 
                                        "R2")
    )

    # Add traces
    rmse = df.groupby('timestamp').apply(lambda x: np.power(mean_squared_error(x['y'], x['preds']), 0.5), include_groups=False).reset_index()
    r2 = df.groupby('timestamp').apply(lambda x: r2_score(x['y'], x['preds']), include_groups=False).reset_index()
    x = df['timestamp'].unique().tolist()

    fig.add_trace(go.Scatter(x=x, y=rmse.values), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=r2.values), row=1, col=2)

    # Update title and height
    fig.update_layout(title_text="<b>Model Performance</b>", 
                      height=400, 
                      width=1200, 
                      paper_bgcolor='#222222', 
                      font={'color':'#dadada'}, 
                      template='plotly_dark',
                      title={'x':0.5, 'font': {'color':'#BB86FC'}},
                      showlegend=False
    )

    return fig

def make_small_plots(df):
    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=4, subplot_titles=("Origin", 
                                        "Amount", 
                                        "Trans L24H", 
                                        "Type")
    )

    # Add traces
    p1 = df['x0'].values
    p2 = df['x1'].values
    p3 = df['x2'].values
    p4 = df['x3'].values

    fig.add_trace(go.Histogram(x=p1, nbinsx=50, opacity=0.6, histnorm='probability density', name='x1 - model build'), row=1, col=1)
    fig.add_trace(go.Histogram(x=p1, nbinsx=50, opacity=0.6, histnorm='probability density', name='x1 - recent'), row=1, col=1)

    fig.add_trace(go.Histogram(x=p2, nbinsx=50, opacity=0.6, histnorm='probability density', name='x2 - model build'), row=1, col=2)
    fig.add_trace(go.Histogram(x=p2, nbinsx=50, opacity=0.6, histnorm='probability density', name='x2 - recent'), row=1, col=2)
    
    fig.add_trace(go.Histogram(x=p3, nbinsx=50, opacity=0.6, histnorm='probability density', name='x3 - model build'), row=1, col=3)
    fig.add_trace(go.Histogram(x=p3, nbinsx=50, opacity=0.6, histnorm='probability density', name='x3 - recent'), row=1, col=3)

    fig.add_trace(go.Histogram(x=p4, nbinsx=50, opacity=0.6, histnorm='probability density', name='x4 - model build'), row=1, col=4)
    fig.add_trace(go.Histogram(x=p4, nbinsx=50, opacity=0.6, histnorm='probability density', name='x4 - recent'), row=1, col=4)

    # Update xaxis properties
    # fig.update_xaxes(title_text="xaxis 1 title", row=1, col=1)
    # fig.update_xaxes(title_text="xaxis 2 title", row=1, col=2)
    # fig.update_xaxes(title_text="xaxis 3 title", row=1, col=3)
    # fig.update_xaxes(title_text="xaxis 4 title", row=1, col=4)

    # # Update yaxis properties
    # fig.update_yaxes(title_text="yaxis 1 title", row=1, col=1)
    # fig.update_yaxes(title_text="yaxis 2 title", row=1, col=2)
    # fig.update_yaxes(title_text="yaxis 3 title", row=1, col=3)
    # fig.update_yaxes(title_text="yaxis 4 title", row=1, col=4)

    # Update title and height
    fig.update_layout(title_text="<b>Feature Monitoring</b>", 
                      height=300, 
                      width=1200, 
                      paper_bgcolor='#222222', 
                      font={'color':'#dadada'}, 
                      template='plotly_dark',
                      title={'x':0.5, 'font': {'color':'#BB86FC'}},
                    #   showlegend=False,
                      barmode='overlay',
                      legend={
                            'orientation':'h',  # Horizontal orientation
                            'yanchor':'bottom',
                            'y':-0.5,  # Adjust this value as needed to position the legend below the subplots
                            'xanchor':'center',
                            'x':0.5
                      }
    )

    return fig

if __name__ == '__main__':
    N_OBS_TO_GENERATE  = 1000
    model_version = 1.0
    dg = DataGenerator()
    df = dg.generate_data(N_OBS_TO_GENERATE)
    preds, new_model_flag, df_new_model_dev, rmse_new = scoring(df)
    df['preds'] = preds

    # Load config dict
    # config = ut.load_config()
    # app.run(host=config['host'], port=config['port'], debug=config['debug'])
    app.run(debug=True)

