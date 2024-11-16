from flask import Flask, render_template, request, jsonify
import ast
import networkx as nx
import pandas as pd
from Graph import Graph  # Assuming your Graph class is in graph.py

app = Flask(__name__)

# Initialize the Graph
movies_file = "movies_metadata.csv"
credits_file = "credits.csv"
movie_graph = Graph(movies_file, credits_file)
movie_graph.read_data()
movie_graph.build_graph()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/visualize', methods=['POST'])
def visualize():
    movie_id = request.form.get('movie_id')
    max_connections = int(request.form.get('max_connections', 15))

    # Visualize the graph for the given movie_id and return JSON data
    graph_data = movie_graph.visualize_graph(movie_id, max_connections)

    return jsonify({'graph_data': graph_data})

if __name__ == '__main__':
    app.run(debug=True)