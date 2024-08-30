import os
import ast
import time 
import json
import logging
import requests

from app.helperfunctions import *
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder

from pymilvus import (utility)
# from pymilvus.decorators import retry_on_rpc_failure

# Set retrytimes to 3
# retry_on_rpc_failure(retry_times = 3)


import dotenv
dotenv.load_dotenv()


# Configure the logging module
logging.basicConfig(level=logging.INFO)



# We have to define a global variable where we are going to store the collection object form milvus
# This has to be done in order to pass the collection object from lifespan events to the other services
#collection_dict = {}


# Lifespan events op app
# When starting: Connects to milvus, loads collection, tries to build index
# When closing app: Releases collection, disconnect
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = milvus_startup()
    yield
    # End application
    milvus_disconnect()



# APP 
app = FastAPI(lifespan=lifespan)

@app.post("/topk")
async def get_top_k_articles(request:Request, topk:int, collection_name : str, return_chunks: bool = False):


    data = await request.json()
    questions = data["questions"]
    n_questions = len(questions)

    INDEXER_HOST = str(os.getenv("INDEXER_HOST"))
    INDEXER_PORT = str(os.getenv("INDEXER_PORT"))

    collection = select_collection(collection_name, level = 3)


    # We perform the embeddings of the questions
    json_in = {"sentences": questions}
    logging.info(f" Performing the embeddings of {n_questions} questions")
    start = time.time()
    response = requests.post(f"http://{INDEXER_HOST}:{INDEXER_PORT}/embed-sentences", json = json_in)
    embedded_questions = response.json()["embeddings"]
    end = time.time()
    logging.info(f" It took {end - start} seconds to perform embed the questions")




    # We prepare search parameters for vector similarity search
    start = time.time()
    search_params = {
                "metric_type": "COSINE", # We select cosine similarity for the search
                }

    logging.info(f" Performing top_{topk} on {n_questions} questions")
    search_results = collection.search(
                data = embedded_questions, 
                anns_field ="embeddings", 
                # the sum of `offset` in `param` and `limit` 
                # should be less than 16384.
                param=search_params,
                limit= topk, # Topk
                expr=None, # Expresion for query
                # set the names of the fields you want to 
                # retrieve from the search result.
                output_fields=['document_id', 'index_range'],
                consistency_level="Strong"
            )
    end = time.time()
    logging.info(f" It took {end - start} seconds to perform the vector search")


    # Preparing the result with some list comprehension
    logging.info(f" Preparing the results of top_{topk} on {n_questions} questions")


    start = time.time()
    # Retrieve all paper contents in a function: 
    all_paper_ids = list(set([hit.entity.get('document_id') for hits in search_results for hit in hits])) # list of all unique paper_ids that were obtained by the vector search 



    if return_chunks == True:
        # Querry all paper contents and save them into a dict
        query_result = get_all_paper_contents(all_paper_ids)
        paper_contents = {id:content for id,content in query_result}

        # In case we couldnt retrieve some paper contents, we will do some logging as caution 

        successful_paper_content_retrieval = list(paper_contents.keys())
        unsuccessful_paper_content_retrieval = [id for id in all_paper_ids if id not in successful_paper_content_retrieval]

        if len(unsuccessful_paper_content_retrieval) != 0: logging.warning(f" Could not retrieve the content of paper ids {unsuccessful_paper_content_retrieval} after vector search")
        
        
        pre_output = { 

            questions[i]:[
                {
                    "document_id": hit.entity.get('document_id'), 
                    "chunk": paper_contents[hit.entity.get('document_id')][ast.literal_eval(hit.entity.get('index_range'))[0]: ast.literal_eval(hit.entity.get('index_range'))[1]],
                    "distance": hit.distance
                } 

            if hit.entity.get('document_id') in successful_paper_content_retrieval
            else 
                {
                    "document_id": hit.entity.get('document_id'), 
                    "chunk": None,
                    "distance": hit.distance
                } 

            for hit in hits
            ]
            
            for i,hits in enumerate(search_results)
                if dict
        }

    else:
        pre_output = { 

            questions[i]:[
                {
                    "document_id": hit.entity.get('document_id'), 
                    "chunk": None,
                    "distance": hit.distance
                } 
            for hit in hits
            ]
            for i,hits in enumerate(search_results)
        }
    
    # Write two functions that take a list of dictionaries 
    out = {
            question: { 
                "ranking": perform_ranking(pre_output[question]),
                "raw_results": pre_output[question]
            } 
    for question in questions}

    end = time.time()
    logging.info(f" It took {end - start} seconds to pepare the result")

    return out

# Calls idnexer to embed a batch of paper contents and insert them into milvus
@app.post("/embedding")
async def embedding_insert(request : Request, collection_name : str, upsert: bool = False):

    json_in = await request.json()
    requested_ids = json_in['id']

    # Select collection according to the desired level of sanity checks 
    level = 1
    if upsert: level = 2
    collection = select_collection(collection_name, level = level)
    

    INDEXER_PORT= str(os.getenv("INDEXER_PORT"))
    INDEXER_HOST = str(os.getenv("INDEXER_HOST"))

    try:  
        logging.info(f" Attempting to embed papers with document_ids: {requested_ids}")
        response = requests.post(f"http://{INDEXER_HOST}:{INDEXER_PORT}/embed-articles", json = json_in)
        logging.info(f" Embedder status quo {response.status_code}")

        json_out = response.json()

        # save the embeddings in milvus
        data = [
            # We change each string into int
            # TODO: Is this really necessary??
            [item for item in json_out["flattened_indexes"]],
            json_out["flattened_indexes_ranges"],
            json_out["embeddings"]
            #[1 for item in json_out["flattened_indexes"]] # Save the index WHY IS THIS HERE?
        ]
        successful_embeddings = json_out['successful_embeddings']
        unsuccessful_embeddings = json_out['unsuccessful_embeddings']

        if len(successful_embeddings) == 0:
            logging.warning(f" No successful embeddings where returned, requested_ids: {requested_ids}") 
            raise HTTPException(status_code = 500, detail = "No successful embeddings where returned" )

    except Exception as e:
        logging.warning(f" Could not perform the embeddings of papers with document_ids: {requested_ids}. error: {e}") 
        raise


    if upsert:
        try:
            # Delete previous embeddings before inserting
            deleted_flag = delete_previous_embeddings(collection = collection, ids = successful_embeddings)
            # Insert embeddings data into collection
            collection.insert(data)
        except Exception as e:
            raise

    else:
        try: 
            # Insert embeddings data into collection
            collection.insert(data)
        except Exception as e:
            raise


    logging.info(" Embedding and milvus insert performed successfully")

    return {"successful_embeddings" : successful_embeddings, "unsuccessful_embeddings": unsuccessful_embeddings}


@app.get("/create-collection")
async def create_collection(collection_name : str):

    # If collection name already exist, dont do anything.

    if utility.has_collection(collection_name):
        return {"message": f"Collection {collection_name} already exist"}

    try: 
        logging.info(f" Creating '{collection_name}' collection")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255), 
            FieldSchema(name="index_range", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="embeddings_version", dtype=DataType.INT64, max_length=255)

            ]
        schema = CollectionSchema(fields, description="Embeddings vectors of chunks of papers with their document_id")
        _ = Collection(name=collection_name, schema = schema, using="default")
        logging.info(f" Collection '{collection_name}' created")
    except Exception as e:
        logging.error(f" Collection '{collection_name}' was not created, details: {e}")
        raise

    return {"message": f"Nice! collection {collection_name} created"}




@app.get("/get-collections")
async def create_collection():
    
    collections = utility.list_collections(timeout = None, using = "default")

    return {"collections": collections}



@app.get("/")
def read_root():
    return {"Hello": "Milvus control is running"}
