from flask import Flask, render_template, request, jsonify
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
    dark_mode = request.form.get('dark_mode') == 'true'
    # Visualize the graph for the given movie_name and return JSON data
    graph_data = movie_graph.visualize_graph(movie_name, max_connections, dark_mode=dark_mode)

    # checks if an error message was returned and returns that message to be displayed
    if graph_data[:9] == '{"error":':
        error_message = graph_data[11:-3]
        return jsonify(error=error_message)

    if isinstance(graph_data, list):  # If it's a list of movies
        return jsonify(graph_data)  # Return the movie list as JSON

    return jsonify({'graph_data': graph_data})
@app.route('/suggestions', methods=['GET'])
def suggestions():
    query = request.args.get('query', '')
    movie_suggestions = get_movie_suggestions(query)
    return jsonify({'suggestions': movie_suggestions})

def get_movie_suggestions(movie_title):
    """
    This function generates movie suggestions based on user input.
    It finds movies with titles containing the entered movie title.
    """
    filtered_movies = movie_graph.movies_df[movie_graph.movies_df['original_title'].str.contains(movie_title, case=False, na=False)]
    # Drop duplicates based on 'original_title' and get the top 5 suggestions
    suggestions = filtered_movies['original_title'].drop_duplicates().tolist()[:5]
    return suggestions

@app.route('/visualizeTwoMovies', methods=['POST'])
def visualizeTwoMovies():
    movie_id = request.form.get('movie_id')
    movie_id2 = request.form.get('movie_id2')
    max_connections = int(request.form.get('max_connections'))
    max_distance = int(request.form.get('max_distance'))
    dark_mode = request.form.get('dark_mode') == 'true'

    # assigns and returns the actual title not the user-typed one
    movie_name = movie_graph.movies_df.loc[movie_graph.movies_df['id'] == movie_id, 'original_title'].iloc[0]
    movie_name2 = movie_graph.movies_df.loc[movie_graph.movies_df['id'] == movie_id2, 'original_title'].iloc[0]

    # Visualize the graph for the given movie_name and return JSON data
    graph_data = movie_graph.visualize_graph_by_id(movie_id, movie_name, max_connections, movie_id2, movie_name2, max_distance, dark_mode=dark_mode)
    # checks if an error message was returned and returns that message to be displayed
    if graph_data[:9] == '{"error":':
        error_message = graph_data[11:-3]
        return jsonify(error=error_message)

    return jsonify({'graph_data': graph_data})

def get_movie_title_by_id(movie_id):
    movie = movie_graph.movies_df.loc[movie_graph.movies_df['id'] == movie_id, 'original_title']
    return movie.iloc[0]


@app.route('/visualizeID', methods=['POST'])
def visualizeID():
    # Get the movie ID from the request
    movie_id = request.form.get('movie_id')
    max_connections = int(request.form.get('max_connections', 15))
    dark_mode = request.form.get('dark_mode') == 'true'

    # Retrieve the original title using the movie_id from your data source
    movie_name = get_movie_title_by_id(movie_id)

    # Call the function to generate the graph data
    graph_data = movie_graph.visualize_graph_by_id(movie_id, movie_name, max_connections, dark_mode=dark_mode)

    # Return the graph data along with the movie name (optional)
    return jsonify({'graph_data': graph_data, 'movie_name': movie_name})

@app.route('/select_movie', methods=['POST'])
def select_movie():
    movie_id = request.form.get('movie_id')
    max_connections = int(request.form.get('max_connections', 15))
    dark_mode = request.form.get('dark_mode') == 'true'

    graph_data = movie_graph.visualize_graph_by_id(movie_id, max_connections, dark_mode=dark_mode)

    return jsonify({'graph_data': graph_data})

@app.route('/get_movie_list', methods=['POST'])
def get_movie_list():
    start_movie_name = request.form.get('start_movie')
    target_movie_name = request.form.get('target_movie')
    start_movie_data = movie_graph.get_movie_list(start_movie_name)
    target_movie_data = movie_graph.get_movie_list(target_movie_name)

    # checks if an error message was returned and returns that message to be displayed
    if start_movie_data[:9] == '{"error":':
        if target_movie_data[:9] == '{"error":' and target_movie_name is not None:
            error_message = start_movie_data[11:-3] + "\n" + target_movie_data[11:-3]
            return jsonify(error=error_message)
        error_message = start_movie_data[11:-3]
        return jsonify(error=error_message)
    if target_movie_data[:9] == '{"error":'and target_movie_name is not None:
        error_message = target_movie_data[11:-3]
        return jsonify(error=error_message)

    return jsonify({'start_movie_data': start_movie_data, 'target_movie_data': target_movie_data})

import time
@app.route('/bfs', methods=['POST'])
def bfs():
    start_movie_name = request.form.get('start_movie')
    target_movie_name = request.form.get('target_movie')
    start_movie_id = request.form.get('start_movie_id')
    target_movie_id = request.form.get('target_movie_id')
    dark_mode = request.form.get('dark_mode')

    # Perform BFS and return the result, while keeping track of time it takes
    start_time = time.time()
    graph_data = movie_graph.find_kevin_bacon_number_bfs(start_movie_name, target_movie_name, start_movie_id, target_movie_id, dark_mode=dark_mode)
    end_time = time.time()
    run_time = (end_time - start_time)
    return jsonify({'graph_data': graph_data, 'run_time': run_time})

@app.route('/dijkstra', methods=['POST'])
def dijkstra():
    start_movie_name = request.form.get('startMovie')
    target_movie_name = request.form.get('targetMovie')
    start_movie_id = request.form.get('start_movie_id')
    target_movie_id = request.form.get('target_movie_id')
    dark_mode = request.form.get('dark_mode')

    # Perform dijkstra and return the result, while keeping track of time it takes
    start_time = time.time()
    graph_data = movie_graph.dijkstra(start_movie_name, target_movie_name, start_movie_id,target_movie_id, dark_mode=dark_mode)
    end_time = time.time()
    run_time = (end_time - start_time)
    return jsonify({'graph_data': graph_data, 'run_time': run_time})

@app.route('/handle-options', methods=['POST'])
def handle_options():
    # Get the selected options from the request
    data = request.get_json()
    selected_options = data.get('selectedOptions', [])

    # Perform any action based on the selected options
    print(f"Selected options: {selected_options}")
    movie_graph.choose_options(selected_options)

    # Return a response
    return jsonify({
        "message": f"You selected: {', '.join(selected_options)}"
    })
if __name__ == '__main__':
    app.run(debug=True)