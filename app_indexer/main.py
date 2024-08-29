import time
import numpy as np
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from app_indexer.helperfunctions import *
import nltk

from fastapi.middleware.cors import CORSMiddleware

import dotenv

dotenv.load_dotenv()

# NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

from sentence_transformers import SentenceTransformer

# Model download and model init
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# NOTE: By default, input text longer than 256 word pieces is truncated.

app = FastAPI()

# Cors configuration
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""

Given a list of paper ids passed as a json file, performs a paragraph-wise embedding their contents and return the fla

"""


@app.post("/embed-articles")
async def embed_articles(request: Request):
    data = await request.json()

    requested_ids = data['id']
    query_result = get_all_paper_contents(requested_ids)

    retrieved_ids = [id for id, content in query_result]
    retrieved_contents = [content for id, content in query_result]

    #Chunk all articles by paragrphs and retrieve indexes
    chunks_and_indexranges = [preprocessing(item, min_len = 400, max_len = 4200) for item in retrieved_contents]

    index_ranges = [ranges for chunks, ranges in chunks_and_indexranges]
    chunked_text = [chunks for chunks, ranges in chunks_and_indexranges]

    # do a chunk count for each article and do a flattening of chunked_text
    chunk_counts = [len(item) for item in chunked_text]
    # flattened array 
    flattened = [x for chunks in chunked_text for x in chunks]

    # flattened_indexes    
    touples = [(retrieved_ids[i], chunk_counts[i]) for i in range(len(retrieved_ids))]  # Touples
    flattened_indexes = [id for id, count in touples for i in range(count)]
    # Flattened index touples and convert touple into str to save it into milvus
    flattened_index_ranges = [f"{idx_tuple.tolist()}" for x in index_ranges for idx_tuple in x]





    # Launch embedding and transform output to python list so that it can be handle by FastAPI
    print(f"info: Embedding {len(flattened)} chunks of text from {len(retrieved_ids)} articles")
    start = time.time()
    embeddings = model.encode(flattened).tolist()  #
    end = time.time()

    print(f"info: It took {end - start} seconds to perform the embeddings")

    successful_embeddings = list(set(flattened_indexes))
    successful_embeddings = [str(item) for item in successful_embeddings]  # Transform ids into strings just in case
    unsuccessful_embeddings = [item for item in requested_ids if item not in successful_embeddings]
    unsuccessful_embeddings = [str(item) for item in unsuccessful_embeddings]  # Transform ids into strings just in case

    out = {
        "successful_embeddings": successful_embeddings,
        "unsuccessful_embeddings": unsuccessful_embeddings, 
        "flattened_indexes": flattened_indexes, 
        "flattened_indexes_ranges": flattened_index_ranges,
        "embeddings": embeddings
        }

    return jsonable_encoder(out)


"""

Given a json containing a batch of sentences inputted in a POST request, returns the embeddigns of the sentences

"""


@app.post("/embed-sentences")
async def embed_sentences(request: Request):
    data = await request.json()
    sentences = data["sentences"]

    print(f"info: Embedding {len(sentences)} sentences")
    start = time.time()
    embeddings = model.encode(sentences).tolist()
    end = time.time()

    print(f"info: It took {end - start} seconds to perform the embeddings")
    out = {"embeddings": embeddings}

    return jsonable_encoder(out)




@app.post("/test-preprocessing")
async def test_preprocessing(request: Request):
    data = await request.json()

    requested_ids = data['id']
    query_result = get_all_paper_contents(requested_ids)
    


    retrieved_ids = [id for id, content in query_result]
    retrieved_contents = [content for id, content in query_result]

    text = retrieved_contents[0]

    chunks, split_idx = preprocessing(text)

    out = {"chunks": chunks.tolist(), "retrieved_chunks": [text[index_range[0]:index_range[1]] for index_range in split_idx], "full_text": text }

    return jsonable_encoder(out)









@app.get("/n_grams/{n}/{id}")
async def n_grams(n: int, id: str):
    text = str(get_paper_content(id))
    n_grams_list = extract_ngrams_from_article(text, n)
    return str(n_grams_list)


# Endpoint sur ID
@app.get("/keywords/{id}")
async def get_keywords(id: str):
    article_content = get_paper_content(id)
    keywords = extract_keywords(article_content[0])
    return {"id": id, "keywords": keywords}


@app.get("/")
def read_root():
    return {"Hello": "The indexer is running"}
