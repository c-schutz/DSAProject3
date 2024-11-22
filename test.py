import app
from Graph import Graph

movies_file = "movies_metadata.csv"
credits_file = "credits.csv"
movie_graph = Graph(movies_file, credits_file)

print(app.get_movie_title_by_id(277834))