# Movie Graph

## Overview

The Movie Graph project allows users to explore relationships between movies based on shared actors. It implements two algorithms: Breadth-First Search (BFS) and Dijkstra's algorithm, to find paths between movies and analyze their connections.

## Features

- **Find connections between movies**: Use BFS to find a path from one movie to another based on shared actors.
- **Shortest path calculation**: Use Dijkstra's algorithm to find the shortest path between two movies, considering edge weights.
- **Visualization**: Visualize the graph and the paths between movies.

## Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `networkx`
  - `plotly`
  - `heapq`
  - `ast`
  - `json`

## Setup

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