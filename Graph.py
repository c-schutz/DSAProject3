import heapq
import networkx as nx
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import os
import ast

class Graph:
    def __init__(self, movies_file, credits_file):
        self.movies_file = movies_file
        self.credits_file = credits_file
        self.movies_df = None
        self.credits_df = None
        self.graph = nx.Graph()
        self.options = []


    def read_data(self, filename="processed_data.pkl"):
        """
        Reads and processes the movie and credits data. If a processed file exists, load it to save time.
        """
        # Check if the preprocessed file exists
        if os.path.exists(filename):
            print(f"Loading preprocessed data from {filename}...")
            self.movies_df = pd.read_pickle(filename)
            return

        # Process data if no preprocessed file exists
        print("Preprocessed file not found. Reading and processing raw data...")
        self.movies_df = pd.read_csv(self.movies_file, low_memory=False)
        self.credits_df = pd.read_csv(self.credits_file, low_memory=False)

        self.movies_df['id'] = self.movies_df['id'].astype(str)
        self.credits_df['id'] = self.credits_df['id'].astype(str)

        self.credits_df['cast'] = self.credits_df['cast'].apply(ast.literal_eval)
        self.credits_df['crew'] = self.credits_df['crew'].apply(ast.literal_eval)

        # Merge movies and credits data
        self.movies_df = self.movies_df.merge(self.credits_df, on='id', how='inner')

        # Save the processed data for future use
        print(f"Saving preprocessed data to {filename}...")
        self.movies_df.to_pickle(filename)

    def save_graph(self, filename="graph.pkl"):
        """
        Save the built graph to a file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved to {filename}")

    def load_graph(self, filename="graph.pkl"):
        """
        Load a graph from a file. Returns True if successful, False otherwise.
        """
        try:
            with open(filename, "rb") as f:
                self.graph = pickle.load(f)
            print(f"Graph loaded from {filename}")
            return True
        except FileNotFoundError:
            print("No saved graph found. Building a new graph.")
            return False

    def build_graph(self, filename="graph.pkl"):
        """
        Build a graph if no saved graph exists; otherwise, load the saved graph.
        """
        # Try to load the graph
        if self.load_graph(filename):
            return  # Graph was successfully loaded, no need to rebuild

        # If no graph exists, build it
        actor_to_movies = {}

        # Create a mapping of actor to movies they appear in
        for _, row in self.movies_df.iterrows():
            movie_id = row['id']
            cast = row['cast']

            for actor in cast:
                actor_name = actor['name']
                if actor_name not in actor_to_movies:
                    actor_to_movies[actor_name] = []
                actor_to_movies[actor_name].append(movie_id)

        # Add edges between movies based on shared actors
        for actor, movies in actor_to_movies.items():
            for i, movie_a in enumerate(movies):
                for movie_b in movies[i + 1:]:
                    if self.graph.has_edge(movie_a, movie_b):
                        self.graph[movie_a][movie_b]['weight'] += 1  # Increment weight
                        self.graph[movie_a][movie_b]['actors'].append(actor)  # Add actor
                    else:
                        self.graph.add_edge(movie_a, movie_b, weight=1, actor=actor, actors=[actor])  # Initialize edge

        # Save the graph after building
        self.save_graph(filename)

    # function to see if there are duplicate movies
    def get_movie_list(self, movie_name):
        if movie_name is None:
            return json.dumps({'error': 'No movie name provided.'})

        # Find matching movies
        matching_movies = self.movies_df[self.movies_df['original_title'].str.lower() == movie_name.lower()]

        if matching_movies.empty:
            return json.dumps({'error': f"Movie '{movie_name}' not found."})

        if len(matching_movies) > 1:
            # Make a list of matching movies for user selection
            movie_data = matching_movies[['original_title', 'id', 'release_date', 'cast']].to_dict(orient='records')

            # Add processed release date and cast names to the dictionary
            for movie in movie_data:
                movie['release_date'] = movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown Year'

            for idx, row in matching_movies.iterrows():
                cast_names = [actor['name'] for actor in row['cast'][:5]]  # Get the top 5 actors' names
                cast_str = ', '.join(cast_names)

                # Find the movie in movie_data by matching IDs or another unique identifier
                for movie in movie_data:
                    if movie['id'] == row['id']:
                        movie['cast'] = cast_str  # Update the cast in the movie data

            # return list of movie data including release date and cast
            return movie_data
        # If a single movie is found, return its ID
        return matching_movies.iloc[0]['id']
    def visualize_graph(self, movie_name=None, max_connections=15):
        if movie_name is None:
            return json.dumps({'error': 'No movie name provided.'})

        # Find matching movies
        matching_movies = self.movies_df[self.movies_df['original_title'].str.lower() == movie_name.lower()]

        if matching_movies.empty:
            return json.dumps({'error': f"Movie '{movie_name}' not found."})
        # if there are duplicate movies gets the data for each movie from other function
        if len(matching_movies) > 1:
            return self.get_movie_list(movie_name)

        # If a single movie is found, get its ID
        movie_id = matching_movies.iloc[0]['id']
        # returns original title from dataset so there's no capitalization errors
        return self.visualize_graph_by_id(movie_id, matching_movies.iloc[0]['original_title'], max_connections)

    def visualize_graph_by_id(self, movie_id=None, movie_title=None, max_connections=15):
        fig = make_subplots()

        if movie_id is not None:
            movie_id = str(movie_id)

        # Use the entire graph if no movie_id is provided, otherwise create a subgraph for the movie and its neighbors
        subgraph = self.graph if movie_id is None else self.graph.subgraph(
            [movie_id] + list(self.graph.neighbors(movie_id)))

        # Limit the number of connections per node
        if movie_id is not None:
            # Get the neighbors and sort by the number of connections (edges)
            neighbors = list(self.graph.neighbors(movie_id))
            neighbors_sorted = sorted(neighbors, key=lambda x: len(list(self.graph.neighbors(x))), reverse=True)

            # Only keep the top `max_connections` neighbors
            limited_neighbors = neighbors_sorted[:max_connections]

            # Create a subgraph with the limited number of neighbors
            subgraph = self.graph.subgraph([movie_id] + limited_neighbors)

        # Layout of the graph
        pos = nx.spring_layout(subgraph, scale=None)
        # Create edges for the plot, using the edge weights
        trace_edges = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#444'),
            hoverinfo='text',
            text=[]
        )

        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = subgraph[edge[0]][edge[1]]['weight']  # Get the weight (number of shared actors)
            shared_actors = ', '.join(subgraph[edge[0]][edge[1]]['actors'])  # Get the shared actors
            trace_edges['x'] += tuple([x0, x1, None])
            trace_edges['y'] += tuple([y0, y1, None])

            # Set the hoverinfo to show the weight and shared actors
            trace_edges['text'] += tuple([f"Shared Actors: {shared_actors}\nWeight (Shared Actors): {weight}"])

        # Create nodes for the plot
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            hovertext=[],
            textposition='top center',
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                color=[],
                colorbar=dict(
                    thickness=25,
                    xanchor='left',
                    titleside='right',
                ),
                line=dict(width=2)
            )
        )

        # Add nodes to the plot
        for node in subgraph.nodes():
            # Fetch the movie name (title) and budget from the movies dataframe
            movie_name = self.movies_df.loc[self.movies_df['id'] == node, 'original_title'].values
            movie_budget = self.movies_df.loc[self.movies_df['id'] == node, 'budget'].values  # Assuming 'rating' column exists
            movie_date = self.movies_df.loc[self.movies_df['id'] == node, 'release_date'].values[0]
            movie_name = movie_name[0] if movie_name.size > 0 else node  # Fallback to node ID if no name found
            budget_value = float(movie_budget[0]) if movie_budget.size > 0 else 0  # Fallback to 0 if no rating found
            year_value = float(movie_date.split('/')[-1]) if len(movie_date) > 0 else 0

            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            if len(self.options) > 0:
                if self.options[0] == "Budget Value":
                    node_trace['marker']['color'] += tuple([budget_value])  # Set color based on rating
                elif self.options[0] == "Date Released":
                    node_trace['marker']['color'] += tuple([year_value])
                else:
                    node_trace['marker']['color'] += tuple([1])
            else:
                node_trace['marker']['color'] += tuple([1])

            # If a base movie is provided, include shared actors in the hover text
            if movie_id and node != movie_id and subgraph.has_edge(movie_id, node):
                shared_actors = ', '.join(subgraph[movie_id][node]['actors'])  # Get shared actors
                node_text = f"Movie: {movie_name}"
                node_hover_text = f"Shared Actors with {movie_title}: {shared_actors}"
            else:
                node_text = f"Movie: {movie_name}"
                node_hover_text = ""

            node_trace['text'] += tuple([node_text])  # Add hover text for the node
            node_trace['hovertext'] += tuple([node_hover_text])

        # Add the traces to the figure
        fig.add_trace(trace_edges)
        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)

        # Return the figure as JSON
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Helper function to disambiguate movie titles
    def disambiguate_movie(self, movie_name):
        matching_movies = self.movies_df[self.movies_df['original_title'].str.lower() == movie_name.lower()]

        if matching_movies.empty:
            print(f"Movie '{movie_name}' not found.")
            return None

        if len(matching_movies) == 1:
            return matching_movies.iloc[0]['id']

        print(f"Multiple movies found with the title '{movie_name}':")
        for idx, row in matching_movies.iterrows():
            cast_names = [actor['name'] for actor in row['cast'][:5]]
            print(
                f"  [{idx}] {row['original_title']} ({row['release_date'] if pd.notna(row['release_date']) else 'Unknown Year'})")
            print(f"      Cast: {', '.join(cast_names)}")

        while True:
            try:
                choice = int(input("Enter the number corresponding to the correct movie: "))
                if choice in matching_movies.index:
                    return matching_movies.loc[choice, 'id']
            except ValueError:
                print("Invalid input. Please enter a number.")
    def find_kevin_bacon_number_bfs(self, start_movie_name, target_movie_name, start_movie_id=None, target_movie_id=None):
        # Disambiguate the start and target movies
        if start_movie_id == None:
            start_movie_id = self.disambiguate_movie(start_movie_name)
        if target_movie_id == None:
            target_movie_id = self.disambiguate_movie(target_movie_name)

        if start_movie_id is None or target_movie_id is None:
            return

        # BFS Initialization
        visited = set()  # Track visited nodes
        queue = [(start_movie_id, [start_movie_id])]  # Queue holds (current_node, path_to_current_node)

        print("Started BFS Search")
        while queue:
            current_node, path = queue.pop(0)

            # If target movie is found, print the path and actors
            if current_node == target_movie_id:
                movie_names = [
                    self.movies_df.loc[self.movies_df['id'] == movie_id, 'original_title'].values[0]
                    for movie_id in path
                ]

                print(f"The Kevin Bacon number from '{start_movie_name}' to '{target_movie_name}' is {len(path) - 1}.")
                print("Path:")

                for i in range(len(path) - 1):
                    movie_a = path[i]
                    movie_b = path[i + 1]
                    shared_actors = self.graph[movie_a][movie_b]['actors']
                    movie_a_name = movie_names[i]
                    movie_b_name = movie_names[i + 1]

                    print(f"  {movie_a_name} -> {movie_b_name} (Shared Actors: {', '.join(shared_actors)})")

                # Visualize the path from start movie to target movie
                subgraph = self.graph.subgraph(path)  # Create subgraph with the BFS path
                return self.visualize_graph_from_subgraph(subgraph)

            # Mark current node as visited
            if current_node not in visited:
                visited.add(current_node)

                # Enqueue all unvisited neighbors
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        print(f"'{target_movie_name}' is not reachable from '{start_movie_name}'.")

    def dijkstra(self, start_movie_name, target_movie_name, start_movie_id=None, target_movie_id=None):
        # Ensure start and target movie IDs are specified
        if start_movie_id == None:
            start_movie_id = self.disambiguate_movie(start_movie_name)
        if target_movie_id == None:
            target_movie_id = self.disambiguate_movie(target_movie_name)

        # Output for improper start/target movie name
        if start_movie_id is None or target_movie_id is None:
            print("Your movie doesn't exist")
            return

        # Initialize distances and previous nodes
        distances = {node: float('inf') for node in self.graph.nodes}
        previous_nodes = {node: None for node in self.graph.nodes}
        distances[start_movie_id] = 0

        # Create a priority queue using heapq
        priority_queue = []
        heapq.heappush(priority_queue, (0, start_movie_id))  # (distance, node)

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            # If we reached the target node, we can stop
            if current_node == target_movie_id:
                break

            # If the distance is greater than the recorded distance, continue
            if current_distance > distances[current_node]:
                continue

            # Iterate over neighbors
            for neighbor in self.graph.neighbors(current_node):
                edge_weight = self.graph[current_node][neighbor]['weight']
                new_distance = distances[current_node] + edge_weight

                # Only consider this new path if it's better
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        # Reconstruct the path
        path = []
        current_node = target_movie_id
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path = path[::-1]

        # Output results
        if distances[target_movie_id] == float('inf'):
            print(f"No path found between '{start_movie_name}' and '{target_movie_name}'.")
        else:
            movie_names = [
                self.movies_df.loc[self.movies_df['id'] == movie_id, 'original_title'].values[0]
                for movie_id in path
            ]
            print(
                f"The shortest path from '{start_movie_name}' to '{target_movie_name}' has a weight of {distances[target_movie_id]:.2f}.")
            print("Path:")
            for i in range(len(path) - 1):
                movie_a = path[i]
                movie_b = path[i + 1]
                shared_actors = self.graph[movie_a][movie_b]['actors']
                movie_a_name = movie_names[i]
                movie_b_name = movie_names[i + 1]
                print(f"  {movie_a_name} -> {movie_b_name} (Shared Actors: {', '.join(shared_actors)})")

            # Visualize the subgraph if needed
            subgraph = self.graph.subgraph(path)
            return self.visualize_graph_from_subgraph(subgraph)

    def visualize_graph_from_subgraph(self, subgraph):
        fig = make_subplots()

        # Layout of the graph
        pos = nx.spring_layout(subgraph, scale=None)

        # Create edges for the plot, using the edge weights
        trace_edges = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#444'),
            hoverinfo='text',
            text=[]
        )

        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = subgraph[edge[0]][edge[1]]['weight']  # Get the weight (number of shared actors)
            shared_actors = ', '.join(subgraph[edge[0]][edge[1]]['actors'])  # Get the shared actors
            trace_edges['x'] += tuple([x0, x1, None])
            trace_edges['y'] += tuple([y0, y1, None])

            # Set the hoverinfo to show the weight and shared actors
            trace_edges['text'] += tuple([f"Shared Actors: {shared_actors}\nWeight (Shared Actors): {weight}"])

        # Create nodes for the plot
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            textposition='top center',
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                color=[],
                colorbar=dict(
                    thickness=25,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right',
                ),
                line=dict(width=2)
            )
        )

        # Add nodes to the plot
        for node in subgraph.nodes():
            # Fetch the movie name (title) from the movies dataframe
            movie_name = self.movies_df.loc[self.movies_df['id'] == node, 'original_title'].values
            movie_name = movie_name[0] if movie_name.size > 0 else node  # Fallback to node ID if no name found

            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['color'] += tuple([len(subgraph.edges(node))])

            # Add hover text for the node
            node_trace['text'] += tuple([f"Movie: {movie_name}"])

        # Add the traces to the figure
        fig.add_trace(trace_edges)
        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    def choose_options(self, new_options):
        self.options = new_options

import time

if __name__ == "__main__":
    full_time = time.time()
    movies_file = "movies_metadata.csv"
    credits_file = "credits.csv"
    movie_graph = Graph(movies_file, credits_file)

    print("Begin Read")
    movie_graph.read_data()
    print("Read Finished")

    print("Begin Build")
    movie_graph.build_graph()
    print("Build Finished")

    # Measure BFS timing
    print("\nRunning BFS:")
    start_time = time.time()
    movie_graph.find_kevin_bacon_number_bfs("Avatar", "Toy Story")
    end_time = time.time()
    print(f"BFS took {end_time - start_time:.4f} seconds.")

    # Measure Dijkstra timing
    print("\nRunning Dijkstra:")
    start_time = time.time()
    movie_graph.dijkstra("Avatar", "Toy Story")
    end_time = time.time()
    print(f"Dijkstra's algorithm took {end_time - start_time:.4f} seconds.")
    finished_time = time.time()

    print(f"Entire main function took {finished_time - full_time:.4f} seconds.")