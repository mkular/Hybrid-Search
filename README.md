# Hybrid-Search
Perform hybrid search over all episode descriptions of a given spotify podcast

## Project Description
This program performs hybrid search over the episode descriptions of a given spotify podcast. 
It uses spotipy to fetch the podcast related data.
Qdrant is used for storing embeddings and their correponding payload/metadate.
Sqlite3 is used for storing chunks in order to perform keyword based search.
Finally streamlit is used to obtain user query and display all results.

![Hybrid Search logic](https://github.com/user-attachments/assets/235f017c-b1ee-4eb0-8ab7-1c28fe234ac5)

## Basic workflow
1. First all the episode data is fetched with a given spotify uri.
2. This is converted into a "episodes.csv" file and stored under data dir i.e. data/episodes.csv
3. The data of episodes.csv is added to qdrant collection called brute_chunks with embedding being performed on individual line obtained from description of episode.
4. We then perform semantic chunking over the chunks obtained via brute force in step 3.
5. These are then stored in Qdrant as semantic_chunks collection.
6. These semantic chunks are also stored in sqlite3 db in order to perform keyword search.
7. Now on streamlit, the user can enter their query.
8. This query is then embedded and matching candidates are fetched via KNN in Qdrant and BM25s in keyword search.
9. These are then re-ranked in order to get the best results.

## Instructions to run[^1]
1. Create a conda env in order to run and install all dependencies.
   <pre>conda create hybrid_search python=3.12
   conda activate hybrid_search
2. Install qdrant container (ensure you have docker installed first) and ensure that the container is up and running.
   please refer: https://qdrant.tech/documentation/guides/installation/ for installation instructions
3. Install all the libraries mentioned in requirements.txt.
   <pre>pip install requirements.txt
4. Modify data/config.yaml with correct directory paths and add the right podcast uri in it.
5. On terminal export your SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.
   <pre> export SPOTIPY_CLIENT_ID="your spotify client id"
    export SPOTIPY_CLIENT_SECRET="your spotify api secret"</pre>
7. Thats it! now run it via:
   <pre> python -m streamlit run hybridsearch.py</pre>
   
  [^1]: __**NOTE**__ : If running for the first time, it will take time to fetch the spotify data, db data to be perared. Please wait for "Search" button to appear in the streamlit UI. Subsequent runs should go through quicker as the Qdrant DB would already have embeddings present.
