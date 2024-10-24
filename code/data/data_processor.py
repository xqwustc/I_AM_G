import os.path
import sys
sys.path.append('/proj/personalized/')

from conf import *
import pandas as pd
import openai
from tqdm import tqdm
import csv

def format_movie_description(row):
    genres = row['genres'].replace('|', ', ')
    return f"Movie {row['title']} with genres {genres}."

def process_movielens():
    data_path = MOVIELENS_IMAGE_PATH
    movid_set = set()
    # Get all movie poster IDs with images based on filenames
    movie_id_list = []
    for filename in os.listdir(data_path):
        base, ext = os.path.splitext(filename)
        if ext.lower() in ['.jpg', '.png']:
            movie_id_list.append(base)
            movid_set.add(base)

    # Exclude movies without images from the user interaction data
    df = pd.read_csv(os.path.join(MOVIELENS_CSV_PATH, 'movies.csv'))

    # Convert movieId column data type to string to match the IDs in movie_id_list
    df['movieId'] = df['movieId'].astype(str)

    # Filter out movies without images
    df = df[df['movieId'].isin(movie_id_list)]

    # Sort user interaction data by user and timestamp (descending) and select the most recent 5 interactions per user
    user_inter = pd.read_csv(os.path.join(MOVIELENS_CSV_PATH, 'ratings.csv'))
    user_inter['movieId'] = user_inter['movieId'].astype(str)
    user_inter = user_inter[user_inter['movieId'].isin(movie_id_list)]
    user_inter = user_inter.sort_values(['userId', 'timestamp'], ascending=[True, False])
    user_inter = user_inter.groupby('userId').head(5)
    user_history_dict = user_inter.groupby('userId')['movieId'].apply(list).to_dict()

    # Reverse the list so the most recent movie appears last
    user_history_dict = {user_id: movie_ids[::-1] for user_id, movie_ids in user_history_dict.items()}

    # Return the interaction data and the set of movies to be used
    return user_history_dict, movid_set

def movie_text_gener():
    history, movie_set = process_movielens()
    df = pd.read_csv(os.path.join(MOVIELENS_CSV_PATH, 'movies.csv'))
    df['movieId'] = df['movieId'].astype(str)
    df['description'] = df.apply(format_movie_description, axis=1)

    # Iterate over all movie IDs in movie_set and print their corresponding descriptions
    movie_des_path = os.path.join(os.path.dirname(MOVIELENS_CSV_PATH), 'movie_descriptions.txt')
    with open(movie_des_path, 'w', encoding='utf-8') as file:
        for movie_id in movie_set:
            # Check if the movie_id exists in the dataframe
            if df['movieId'].isin([movie_id]).any():
                row = df[df['movieId'] == movie_id].iloc[0]
                description = f"{movie_id}: {row['description']}"
            else:
                print(f"{movie_id}: No description available")

            # Write the description to the file
            file.write(description + "\n")
            file.flush()

def image_description_trunker(dataset='movielens'):
    if dataset == 'movielens':
        data_path = MOVIELENS_PATH
        description_path = os.path.join(data_path, 'image_descriptions_origin.txt')

        descriptions_dict = {}

        with open(description_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                assert len(parts) >= 2, "Incomplete description!"
                key = parts[0].strip()
                value = ':'.join(parts[1:]).strip()
                value = ' '.join(value.split())  # Remove extra spaces
                assistant_idx = value.find('ASSISTANT')
                assert assistant_idx != -1, "Incomplete description (no assistant response)!"
                value = value[assistant_idx + len('ASSISTANT') + 1:].strip()
                descriptions_dict[key] = value
        get_keywords_by_gpt(descriptions_dict, dataset)

    elif dataset == 'POG':
        description_path = "/proj/personalized/model/POG_image_descriptions.txt"

        descriptions_dict = {}

        with open(description_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                assert len(parts) >= 2, "Incomplete description!"
                key = parts[0].strip()
                value = ':'.join(parts[1:]).strip()
                value = ' '.join(value.split())
                assistant_idx = value.find('ASSISTANT')
                assert assistant_idx != -1, "Incomplete description (no assistant response)!"
                value = value[assistant_idx + len('ASSISTANT') + 1:].strip()
                descriptions_dict[key] = value

        get_keywords_by_gpt(descriptions_dict, dataset)

def image_description_maker(dataset='POG'):
    if dataset == 'POG':
        description_path = "/proj/personalized/model/POG_image_descriptions.txt"

        descriptions_dict = {}

        with open(description_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                assert len(parts) >= 2, "Incomplete description!"
                key = parts[0].strip()
                value = ':'.join(parts[1:]).strip()
                value = ' '.join(value.split())
                assistant_idx = value.find('ASSISTANT')
                assert assistant_idx != -1, "Incomplete description (no assistant response)!"
                value = value[assistant_idx + len('ASSISTANT') + 1:].strip()
                descriptions_dict[key] = value

        with open(os.path.join(os.path.dirname(POG_PATH), "POG_text.txt"), 'a') as file:
            for ids, des in descriptions_dict.items():
                file.write(f"{ids}: {des}\n")
                file.flush()

def get_file_prefixes(directory):
    prefixes = set()
    for filename in os.listdir(directory):
        prefix = filename.split('.')[0]
        prefixes.add(prefix)
    return prefixes

if __name__ == '__main__':
    process_movielens()
