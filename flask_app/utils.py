import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score
import os
import requests

RMSE_RETRAIN_THRESHOLD = 40.0
N_RECENT_OBS = 1000
ML_INFO_API_URL = 'http://localhost:5002/get-model-info'
ML_RETRAIN_API_URL = 'http://localhost:5002/retrain'

def get_model_info():
    """
    Make an API call to the ML Model service to get model info.
    """
    # Fetch the ML_API_URL environment variable, with a fallback in case it's not set
    ml_api_url = os.environ.get('ML_INFO_API_URL', ML_INFO_API_URL)
    
    # Make a POST request to the predict endpoint
    response = requests.get(ml_api_url)

    model_info = None
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the data from the response
        model_info = response.json()
        print("Success: Model info request succeeded")
    else:
        print("Failed to get model info. Status code:", response.status_code, "Response:", response.text)

    return model_info

def make_retrain_request():
    """
    Make an API call to the ML Model service to retrain the model.
    """
    # Fetch the ML_API_URL environment variable, with a fallback in case it's not set
    ml_api_url = os.environ.get('ML_RETRAIN_API_URL', ML_RETRAIN_API_URL)
    
    # Make a POST request to the predict endpoint
    response = requests.post(ml_api_url)

    # Check if the request was successful
    if response.status_code == 200:
        print("Success: retrain request succeeded")
    else:
        print("Failed to make retrain request. Status code:", response.status_code, "Response:", response.text)

def check_retrain(df) -> None:
    """
    Check if model needs to be retrained.
    """

    # Get recent data
    y = df['y'].iloc[-N_RECENT_OBS:]
    preds = df['y_pred'].iloc[-N_RECENT_OBS:]

    # Calculate rmse
    rmse_current = np.power(mean_squared_error(y, preds), 0.5)

    # If RMSE is too high, retrain model
    print(f'current RMSE: {rmse_current}')
    if rmse_current > RMSE_RETRAIN_THRESHOLD:
        print(f'retraining...')
        make_retrain_request()
        
def create_plot_data(df, model_info) -> defaultdict:
    """
    Create data to be used in plots.
    """

    plot_data = defaultdict(dict)
    group_df = df.groupby('timestamp').agg({'x0':'count', 'y': 'mean', 'y_pred':'mean'}).reset_index()
    x = group_df['timestamp'].values.tolist()
    y1 = group_df['x0'].values.tolist()
    y2 = group_df['y'].values.tolist()
    y3 = group_df['y_pred'].values.tolist()

    plot_data['main'] = {
        'x': [x, x, x],
        'y': [y1, y2, y3]
    }

    rmse = df.groupby('timestamp').apply(lambda x: np.power(mean_squared_error(x['y'], x['y_pred']), 0.5), include_groups=False).reset_index()
    r2 = df.groupby('timestamp').apply(lambda x: r2_score(x['y'], x['y_pred']), include_groups=False).reset_index()

    plot_data['model_perf'] = {
        'x':[x, x],
        'y':[rmse[0].values.tolist(), r2[0].values.tolist()]
    }

    p1_1 = df['x0'].iloc[:-N_RECENT_OBS].value_counts().sort_index()
    p1_2 = df['x0'].iloc[-N_RECENT_OBS:].value_counts().sort_index()

    p2_1 = df['x1'].iloc[:-N_RECENT_OBS].value_counts().sort_index()
    p2_2 = df['x1'].iloc[-N_RECENT_OBS:].value_counts().sort_index()

    p3_1 = df['x2'].iloc[:-N_RECENT_OBS].value_counts().sort_index()
    p3_2 = df['x2'].iloc[-N_RECENT_OBS:].value_counts().sort_index()

    p4_1 = df['x3'].iloc[:-N_RECENT_OBS].value_counts().sort_index()
    p4_2 = df['x3'].iloc[-N_RECENT_OBS:].value_counts().sort_index()

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

    plot_data['model_update_values'] = {}
    plot_data['model_update_values']['version'] = model_info['model_version']
    plot_data['model_update_values']['update_time'] = model_info['training_date_time']
    plot_data['model_update_values']['rmse'] = model_info['rmse']
    
    return plot_data

def make_main_plot(plot_data):
    """
    Create plotly price charts.
    """
    
    # Unpack plot data
    x = plot_data['main']['x'][0]
    y1 = plot_data['main']['y'][0]
    y2 = plot_data['main']['y'][1]
    y3 = plot_data['main']['y'][2]

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y1, name='Volume', opacity=0.7))
    fig.add_trace(go.Scatter(x=x, y=y2, name='Actual Amount', yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=y3, name='Predicted Amount', yaxis="y2", line={'dash':'dash'}))
    fig.update_layout(width=1200, 
                        height=500, 
                        # yaxis_range=[0, 16000], 
                        title_text="<b>Transaction Amounts and Volumes</b>", 
                        paper_bgcolor='#222222', 
                        font={'color':'#dadada'}, 
                        template='plotly_dark',
                        title={'x':0.5, 
                                'font': {'color':'#dadada'}},
                        yaxis={'title': 'Volume'},
                        yaxis2={
                                'title':'Trans Amount',
                                'overlaying':'y',  # This places y2 on top of y
                                'side':'right'  # This places y2 on the right side
                        },
                        legend={
                            'orientation':'h',  # Horizontal orientation
                            'yanchor':'bottom',
                            'y':-0.3,  # Adjust this value as needed to position the legend below the subplots
                            'xanchor':'center',
                            'x':0.5
                        }
    )
    return fig

def make_performance_plots(plot_data):
    """
    Make model performance plots.
    """

    # Unpack plot data
    x = plot_data['model_perf']['x'][0]
    rmse = plot_data['model_perf']['y'][0]
    r2 = plot_data['model_perf']['y'][1]

    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("RMSE", 
                                        "R2")
    )

    fig.add_trace(go.Scatter(x=x, y=rmse), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=r2), row=1, col=2)

    # Update title and height
    fig.update_layout(title_text="<b>Model Performance</b>", 
                      height=400, 
                      width=1200, 
                      paper_bgcolor='#222222', 
                      font={'color':'#dadada'}, 
                      template='plotly_dark',
                      title={'x':0.5, 'font': {'color':'#dadada'}},
                      showlegend=False
    )

    return fig

def make_feature_plots(df):
    """
    Make model feature plots. Feed the raw df to begin with, and then update with
    plot_data dict thereafter.
    """

    # Generate plot data
    x0_0 = df['x0'].iloc[:-N_RECENT_OBS].values
    x0_1 = df['x0'].iloc[-N_RECENT_OBS:].values
    x1_0 = df['x1'].iloc[:-N_RECENT_OBS].values
    x1_1 = df['x1'].iloc[-N_RECENT_OBS:].values
    x2_0 = df['x2'].iloc[:-N_RECENT_OBS].values
    x2_1 = df['x2'].iloc[-N_RECENT_OBS:].values
    x3_0 = df['x3'].iloc[:-N_RECENT_OBS].values
    x3_1 = df['x3'].iloc[-N_RECENT_OBS:].values


    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=4, subplot_titles=("Origin", 
                                        "Amount", 
                                        "Trans L24H", 
                                        "Type")
    )

    # Add traces
    fig.add_trace(go.Histogram(x=x0_0, nbinsx=50, opacity=0.6, histnorm='probability density', name='x1 - model build'), row=1, col=1)
    fig.add_trace(go.Histogram(x=x0_1, nbinsx=50, opacity=0.6, histnorm='probability density', name='x1 - recent'), row=1, col=1)

    fig.add_trace(go.Histogram(x=x1_0, nbinsx=50, opacity=0.6, histnorm='probability density', name='x2 - model build'), row=1, col=2)
    fig.add_trace(go.Histogram(x=x1_1, nbinsx=50, opacity=0.6, histnorm='probability density', name='x2 - recent'), row=1, col=2)
    
    fig.add_trace(go.Histogram(x=x2_0, nbinsx=50, opacity=0.6, histnorm='probability density', name='x3 - model build'), row=1, col=3)
    fig.add_trace(go.Histogram(x=x2_1, nbinsx=50, opacity=0.6, histnorm='probability density', name='x3 - recent'), row=1, col=3)

    fig.add_trace(go.Histogram(x=x3_0, nbinsx=50, opacity=0.6, histnorm='probability density', name='x4 - model build'), row=1, col=4)
    fig.add_trace(go.Histogram(x=x3_1, nbinsx=50, opacity=0.6, histnorm='probability density', name='x4 - recent'), row=1, col=4)

    # Update title and height
    fig.update_layout(title_text="<b>Feature Monitoring</b>", 
                      height=300, 
                      width=1200, 
                      paper_bgcolor='#222222', 
                      font={'color':'#dadada'}, 
                      template='plotly_dark',
                      title={'x':0.5, 'font': {'color':'#dadada'}},
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
