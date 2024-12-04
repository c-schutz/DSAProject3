# Movie Visualization Graph

## Overview

The Movie Vizualization Graph project allows users to explore relationships between movies based on shared actors. It implements two algorithms: Breadth-First Search (BFS) and Dijkstra's algorithm, to find paths between movies and analyze their connections.

## Features

- **Find connections between movies**: Use BFS to find a path from one movie to another based on shared actors.
- **Shortest path calculation**: Use Dijkstra's algorithm to find the shortest path between two movies, considering edge weights.
- **Visualization**: Visualize the graph and the paths between movies.

## Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `networkx`
  - `plotly (version 5.24.1)`
  - `heapq`
  - `ast`
  - `json`

## Version Control
  
    If you encounter issues, your package versions may be incorrect. 
    Make sure that they match what we used below (in white).

![image](https://github.com/c-schutz/DSAProject3/blob/main/img.png?raw=true)
1. **Clone the repository**:
   ```bash
   git clone https://github.com/c-schutz/DSAProject3.git
   cd movie-graph
2. **Install Required Libraries**:
    ```bash
   pip install pandas networkx plotly
3. **Prepare the Data Files**:<br />
    Make sure that both `movies_metadata.csv` and `credits.csv` are in the repo
## Usage
1. **Run the script**:<br />
    Execute `app.py` to start the application
    ```bash
   python app.py
2. **Open the website**:<br />
    After a few minutes(if running for the first time) a website link will appear.<br />
    Looks something like this `http://127.0.0.1:5000`. Click it.
3. **Use the website**:<br />
    Once the website loads your website should look something like this: ![image](https://github.com/user-attachments/assets/e6497ec6-c41f-47ea-93fb-d3e8414cbba6)
4. **Different features**:<br />
    -**Graph Visaulization**:<br />
        To use the `Graph Visualization` type in a movie next to `Enter Movie Title`<br />
        You can then choose the number of connections that the graph will show (starting is 15), but can go as high as you like.<br />
        Then click `Visualize` and watch the magic happen<br />
   -**Breadth-First Search**:<br />
       To use `Breadth-First Search` go to `Choose a Feature` in the top left and select `Breadth-First Serach`<br />
       Enter the start movie title and target movie title and then click `Find Kevin Bacon Number` and watch the magic happen<br />
   -**Dijkstra Search**:<br />
       To use `Dijkstra Search` go to `Choose a Feature` in the top left and select `Dijkstra Search`<br />
       Enter the start movie title and target movie title and then click `Find Kevin Bacon Number` and as always watch the magic happen<br />
   -**Add Second Movie**:<br />
       If you would like to see two movies and all the connections at a certain distance click the `Add Second Movie` button<br />
       Then Enter both Movie Title and Second Movie Title. To change the distance that the BFS will look to connect Movies increase or decrease `Max Distance`<br />
       Finally click `Visualize Two Movies` and watch the magic happen. If the Movies do not share a movie connection this will not work however<br />
   -**Add more data**:<br />
       If you would like to see some different movie data such as `Date Released` or `Revenue` click on the big button in the top middle titled `Choose one or more options`<br />
       Choose as few or as many of the different options as you would like. Then choose the method you would like to look at the movies from and as always watch the magic happen
       
