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
    movie_name = request.form.get('movie_id')
    max_connections = int(request.form.get('max_connections', 15))

    # Visualize the graph for the given movie_name and return JSON data
    graph_data = movie_graph.visualize_graph(movie_name, max_connections)

    if isinstance(graph_data, list):  # If it's a list of movies
        return jsonify(graph_data)  # Return the movie list as JSON

    return jsonify({'graph_data': graph_data})


def get_movie_title_by_id(movie_id):
    movie = movie_graph.movies_df.loc[movie_graph.movies_df['id'] == movie_id, 'original_title']
    return movie.iloc[0]


@app.route('/visualizeID', methods=['POST'])
def visualizeID():
    # Get the movie ID from the request
    movie_id = request.form.get('movie_id')
    max_connections = int(request.form.get('max_connections', 15))

    # Retrieve the original title using the movie_id from your data source
    movie_name = get_movie_title_by_id(movie_id)

    # Call the function to generate the graph data
    graph_data = movie_graph.visualize_graph_by_id(movie_id, movie_name, max_connections)

    # Return the graph data along with the movie name (optional)
    return jsonify({'graph_data': graph_data, 'movie_name': movie_name})

@app.route('/select_movie', methods=['POST'])
def select_movie():
    movie_id = request.form.get('movie_id')
    max_connections = int(request.form.get('max_connections', 15))

    # Visualize the graph for the selected movie ID
    graph_data = movie_graph.visualize_graph_by_id(movie_id, max_connections)

    return jsonify({'graph_data': graph_data})

@app.route('/bfs', methods=['POST'])
def bfs():
    start_movie_name = request.form.get('start_movie')
    target_movie_name = request.form.get('target_movie')

    # Perform BFS and return the result
    graph_data = movie_graph.find_kevin_bacon_number_bfs(start_movie_name, target_movie_name)
    return jsonify({'graph_data': graph_data})

@app.route('/dijkstra', methods=['POST'])
def dijkstra():
    start_movie_name = request.form.get('start_movie')
    target_movie_name = request.form.get('target_movie')

    # Perform BFS and return the result
    graph_data = movie_graph.dijkstra(start_movie_name, target_movie_name)
    return jsonify({'graph_data': graph_data})
if __name__ == '__main__':
    app.run(debug=True)