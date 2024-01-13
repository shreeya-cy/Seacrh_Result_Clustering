from flask import Flask, render_template, request
import plotly
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
import demo

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    k = request.form['k']
    query = request.form['query']

    # Assume get_clusters is defined elsewhere and returns a Plotly figure
    relevant_documents = demo.index_search(query)
    fig = demo.get_clusters(int(k), relevant_documents)

    # Convert the figure to JSON
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    print("FIG:",fig_json)

    return render_template('clusters.html', fig_json=fig_json)

if __name__ == '__main__':
    app.run(debug=True)
