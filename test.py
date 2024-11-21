import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Load the dataset
df = pd.read_csv('movies_metadata.csv', low_memory=False)

poster_paths = df['poster_path']

base_url = 'https://image.tmdb.org/t/p/w500'
for path in poster_paths:
    if pd.notna(path):  # Check if the path is not NaN
        full_url = base_url + path
        try:
            response = requests.get(full_url)
            response.raise_for_status()  # Raise an error for bad responses
            img = Image.open(BytesIO(response.content))
            plt.imshow(img)
            plt.axis('off')  # Turn off axis labels
            plt.show()
            break  # Stop after displaying the first valid image
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except Image.UnidentifiedImageError:
            print(f"Could not identify image at {full_url}")