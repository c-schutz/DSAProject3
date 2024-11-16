import ast
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    def visualize_graph(self, movie_id=None, max_connections=15):
        fig = make_subplots()

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
        pos = nx.spring_layout(subgraph, scale=2)

        # Create edges for the plot, using the edge weights
        trace_edges = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            mode='lines',
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
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
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
                node_hover_text = f"Movie: {movie_name}\nShared Actors with {movie_name}: {shared_actors}"
            else:
                node_hover_text = f"Movie: {movie_name}"

            node_trace['text'] += tuple([node_hover_text])  # Add hover text for the node

        # Add the traces to the figure
        fig.add_trace(trace_edges)
        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
        fig.show()


if __name__ == "__main__":
    movies_file = "movies_metadata.csv"
    credits_file = "credits.csv"

    movie_graph = Graph(movies_file, credits_file)

    movie_graph.read_data()

    movie_graph.build_graph()
    print("done building")
    movie_id = "18"
    movie_graph.visualize_graph(movie_id=movie_id, max_connections=10)
