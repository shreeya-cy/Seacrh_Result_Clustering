from flask import Flask, render_template, request
import plotly
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json
import demo
from dash import Dash, dcc, html, Input, Output

app = Flask(__name__)
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Sample layout for Dash app
dash_app.layout = html.Div([
    dcc.Graph(id='cluster-plot'),
    html.H1(id='hovered-text')
])

@app.route('/')
def index():
    return render_template('index.html')

@dash_app.callback(
    Output('hovered-text', 'children'),
    [Input('cluster-plot', 'hoverData')]
)
def update_hovered_text(hover_data):
    if hover_data is not None and 'points' in hover_data:
        point_data = hover_data['points'][0]
        doc_id = point_data.get('customdata', {}).get('Content', None)
        if doc_id is not None:
            return f"{hover_data}"
        
    return "Hover over a point to see the document information"

@app.route('/cluster', methods=['POST'])
def cluster():
    k = request.form['k']
    query = request.form['query']

    # Assume get_clusters is defined elsewhere and returns a Plotly figure
    relevant_documents = demo.index_search(query)
    fig, docs = demo.get_clusters(int(k), relevant_documents)

    # Convert the figure to JSON
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    dash_app.layout = html.Div([
        dcc.Graph(id='cluster-plot', figure=fig),  # Set the figure directly
        html.H1(id='hovered-text')
    ])

    return render_template('clusters.html', fig_json=fig_json)

if __name__ == '__main__':
    app.run(debug=True)
