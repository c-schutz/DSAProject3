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
        """
        actor_to_movies = {}


        for _, row in self.movies_df.iterrows():
            movie_id = row['id']
            cast = row['cast']

            for actor in cast:
                actor_name = actor['name']
                if actor_name not in actor_to_movies:
                    actor_to_movies[actor_name] = []
                actor_to_movies[actor_name].append(movie_id)


        for actor, movies in actor_to_movies.items():
            for i, movie_a in enumerate(movies):
                for movie_b in movies[i + 1:]:
                    self.graph.add_edge(movie_a, movie_b, actor=actor)

    def visualize_graph(self, movie_id=None):
        fig = make_subplots()

        subgraph = self.graph if movie_id is None else self.graph.subgraph(
            [movie_id] + list(self.graph.neighbors(movie_id)))

        pos = nx.spring_layout(subgraph, scale=2)
        trace_edges = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            trace_edges['x'] += tuple([x0, x1, None])
            trace_edges['y'] += tuple([y0, y1, None])

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
                line=dict(width=2)))

        for node in subgraph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['marker']['color'] += tuple([len(subgraph.edges(node))])
            node_trace['text'] += tuple([f"{node}"])

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

    movie_id = "862"
    movie_graph.visualize_graph()
