from flask import Flask, render_template, request, jsonify
import plotly
import json
import os
import sys
from utils import get_model_info, create_plot_data, make_main_plot, make_performance_plots, make_feature_plots, check_retrain

# Hack to use relative imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../shared_utils'))
sys.path.append(parent_dir)

# Now you can import the module from the parent directory
from db_utils import retrieve_data

app = Flask(__name__)

@app.route('/trans-pred-dashboard')
def index():

    # Get data from the db
    df = retrieve_data()
    df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%d%m %H:%M:%S'))

    model_info = get_model_info()

    plot_data = create_plot_data(df, model_info)

    # Create plots
    main_plot = make_main_plot(plot_data)
    main_plot_json = json.dumps(main_plot, cls=plotly.utils.PlotlyJSONEncoder)

    model_perf = make_performance_plots(plot_data)
    model_perf_json = json.dumps(model_perf, cls=plotly.utils.PlotlyJSONEncoder)

    small_plots = make_feature_plots(df)
    small_plots_json = json.dumps(small_plots, cls=plotly.utils.PlotlyJSONEncoder)

    # Render template
    return render_template('index.html', 
                           main_plot_json=main_plot_json, 
                           model_perf_json=model_perf_json,
                           small_plots_json=small_plots_json,
                           model_info=model_info)

@app.route('/get-new-data')
def get_new_data():
    # Generate new data
    df = retrieve_data()
    df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%d%m %H:%M:%S'))

    # Check for retraining
    check_retrain(df)

    model_info = get_model_info()
    plot_data = create_plot_data(df, model_info)

    return jsonify(plot_data)

if __name__ == '__main__':
    # Load config dict
    # config = ut.load_config()
    # app.run(host=config['host'], port=config['port'], debug=config['debug'])
    app.run(host='0.0.0.0', port='5000', debug=True)

