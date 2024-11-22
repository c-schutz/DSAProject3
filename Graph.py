import ast
from collections import defaultdict

import networkx as nx
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from PriorityQueue import PriorityQueue as pq


class Graph:
    def __init__(self, movies_file, credits_file):
        self.movies_file = movies_file
        self.credits_file = credits_file
        self.movies_df = None
        self.credits_df = None
        self.graph = nx.Graph()

    def read_data(self):
        self.movies_df = pd.read_csv(self.movies_file, low_memory=False)
        self.credits_df = pd.read_csv(self.credits_file, low_memory=False)

        self.movies_df['id'] = self.movies_df['id'].astype(str)
        self.credits_df['id'] = self.credits_df['id'].astype(str)


        self.credits_df['cast'] = self.credits_df['cast'].apply(ast.literal_eval)
        self.credits_df['crew'] = self.credits_df['crew'].apply(ast.literal_eval)

        self.movies_df = self.movies_df.merge(self.credits_df, on='id', how='inner')


    def build_graph(self):
        """
        Build a graph where movies are connected if they share actors.
        Each edge is weighted by the number of actors they share, and
        the list of actors is stored for hover information.
        """
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

        # Add edges between movies based on shared actors, and weight the edges by the number of shared actors
        for actor, movies in actor_to_movies.items():
            for i, movie_a in enumerate(movies):
                for movie_b in movies[i + 1:]:
                    if self.graph.has_edge(movie_a, movie_b):
                        self.graph[movie_a][movie_b]['weight'] += 1  # Increment weight if an edge already exists
                        self.graph[movie_a][movie_b]['actors'].append(actor)  # Add actor to the shared list
                    else:
                        self.graph.add_edge(movie_a, movie_b, weight=1, actor=actor,
                                            actors=[actor])  # Initialize weight and actors

    def visualize_graph(self, movie_name=None, max_connections=15):
        if movie_name is None:
            return json.dumps({'error': 'No movie name provided.'})

        # Find matching movies
        matching_movies = self.movies_df[self.movies_df['original_title'].str.lower() == movie_name.lower()]

        if matching_movies.empty:
            return json.dumps({'error': f"Movie '{movie_name}' not found."})

        if len(matching_movies) > 1:
            # Return the list of matching movies for user selection
            return matching_movies[['original_title', 'id']].to_dict(orient='records')

        # If a single movie is found, get its ID
        movie_id = matching_movies.iloc[0]['id']
        return self.visualize_graph_by_id(movie_id, movie_name, max_connections)

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

            # If a base movie is provided, include shared actors in the hover text
            if movie_id and node != movie_id and subgraph.has_edge(movie_id, node):
                shared_actors = ', '.join(subgraph[movie_id][node]['actors'])  # Get shared actors
                node_hover_text = f"Movie: {movie_name}\nShared Actors with {movie_title}: {shared_actors}"
            else:
                node_hover_text = f"Movie: {movie_name}"

            node_trace['text'] += tuple([node_hover_text])  # Add hover text for the node

        # Add the traces to the figure
        fig.add_trace(trace_edges)
        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, showticklabels=False)

        # Return the figure as JSON
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def find_kevin_bacon_number_bfs(self, start_movie_name, target_movie_name):
        # Helper function to disambiguate movie titles
        def disambiguate_movie(movie_name):
            matching_movies = self.movies_df[self.movies_df['original_title'] == movie_name]

            if matching_movies.empty:
                print(f"Movie '{movie_name}' not found.")
                return None

            if len(matching_movies) == 1:
                return matching_movies.iloc[0]['id']

            print(f"Multiple movies found with the title '{movie_name}':")
            for idx, row in matching_movies.iterrows():
                cast_names = [actor['name'] for actor in row['cast'][:5]]  # Show up to 5 actors for clarity
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

        # Disambiguate the start and target movies
        start_movie_id = disambiguate_movie(start_movie_name)
        target_movie_id = disambiguate_movie(target_movie_name)

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

    def dijkstra(self, start_movie_name, target_movie_name):

        def disambiguate_movie(movie_name):
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

        #make sure start and target movie_id's are proper and specified
        start_movie_id = disambiguate_movie(start_movie_name)
        target_movie_id = disambiguate_movie(target_movie_name)

        #output for improper start/target movie name
        if start_movie_id is None or target_movie_id is None:
            print("Your movie doesn't exist")
            return

        #initialize distances
        distances = {node: float('inf') for node in self.graph.nodes}#d[v]
        previous_nodes = {node: None for node in self.graph.nodes}#p[v]
        distances[start_movie_id] = 0

        #create priority queue
        priority_queue = pq()
        priority_queue.insert((0, start_movie_id))

        found = False
        while not priority_queue.checkEmpty():
            current_distance, current_node = priority_queue.pop()
            adjacent_movies = self.graph.neighbors(current_node)
            if current_node == target_movie_id:
                break
            # if found:
            #     break
            for node in adjacent_movies:
                edge_weight = self.graph[current_node][node]['weight']
                if distances[node] > distances[current_node] + edge_weight:
                    distances[node] = distances[current_node] + edge_weight
                    previous_nodes[node] = current_node
                    # if node == target_movie_id:
                    #     found = True
                    #     break
                    priority_queue.insert((edge_weight, node))

        path = []
        current_node = target_movie_id
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path = path[::-1]

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


if __name__ == "__main__":
    movies_file = "movies_metadata.csv"
    credits_file = "credits.csv"
    # movie_graph = Graph(movies_file, credits_file)
    #
    # print("Begin Read")
    # movie_graph.read_data()
    # print("Read Finished")
    #
    # print("Begin Build")
    # movie_graph.build_graph()
    # print("Build Finished")

    # movie_graph.visualize_graph("862", 15)
    # movie_graph.find_kevin_bacon_number_bfs("Minions", "Devil in a Blue Dress")
    # movie_graph.dijkstra("Minions", "Devil in a Blue Dress")