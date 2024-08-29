import os
import time
import json
import pymysql
import pandas as pd
import numpy as np
import logging
import requests
import dotenv


dotenv.load_dotenv()



from pymilvus import (
    connections,
    db,
    utility,
    Role,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)



# Configure the logging module
logging.basicConfig(level=logging.INFO)

# Connect to milvus within docker app
def milvus_connect():
    connections.connect(
        alias=str(os.getenv("MILVUS_ALIAS")),
        user=str(os.getenv("MILVUS_USERNAME")),
        password=str(os.getenv("MILVUS_PASSWORD")),
        host="standalone",
        port=int(os.getenv("MILVUS_PORT"))#,
        #db_name=str(os.getenv("MILVUS_DB")) # Check if this can be removed
        )

# Connects to milvus and loads collection
def milvus_startup():

    # Connect for first time and create an user
    logging.info(" Attempting to execute Milvus initialization")

    #initialize_milvus()

    # Connect with blends user
    milvus_connect()

    try:
        logging.info("Current users and roles in Milvus")
        logging.info(utility.list_users(include_role_info = True, using="default"))
        logging.info(utility.list_roles(include_user_info = True, using="default"))
    except: pass

    database_name = str(os.getenv("MILVUS_DB"))
    collection_name = str(os.getenv("MILVUS_COLLECTION"))

    # Creates database embeddings if it does not exist
    if database_name not in db.list_database():
        logging.info(f" '{database_name}' database not found, creating it... ")
        db.create_database(database_name)
        logging.info(f" Database '{database_name}' was created ")

    # Switch to embeddings database
    db.using_database(database_name)
    logging.info(f" Switched to '{database_name}' database")

    if utility.has_collection(collection_name) == False:

        logging.info(f" Creating '{collection_name}' collection")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255), 
            FieldSchema(name="index_range", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
        schema = CollectionSchema(fields, description="Embeddings vectors of chunks of papers with their document_id")
        _ = Collection(name=collection_name, schema = schema, using="default")
        logging.info(f" Collection '{collection_name}' created")
        
    # Switch to connection 
    collection = Collection(collection_name)
    logging.info(f" Collection '{collection_name}' selected")

    return None


def initialize_milvus():

    startup_config = False

    # If we manage to stablish a connection to milvus with super user we run the startup
    
    try: 
        connections.connect(
            alias=str(os.getenv("MILVUS_ALIAS")),
            user=str(os.getenv("MILVUS_USER")),
            password=str(os.getenv("MILVUS_PASSWORD")),
            host="standalone",
            port=int(os.getenv("MILVUS_PORT"))
            )
        
        # Change this config to create an account
        startup_config = False

        logging.info(f" Welcome to Milvus")
        logging.info(utility.list_users(include_role_info = True, using="default"))
        logging.info(utility.list_roles(include_user_info = True, using="default"))

        try: 
            # Delete previous blends user to recreate it 
            utility.delete_user(str(os.getenv("MILVUS_USERNAME")), using="default")
        except: pass

    except:
        pass
    
    startup_message = " Milvus initialization SKIPPED"

    if startup_config:

        logging.info(f" Launching Milvus initialization")
        try:     

            logging.info(" Creating "+ str(os.getenv("MILVUS_USERNAME")) +" user")

            role = Role("admin", using="default")
            # Create a role and grand privileges
            #role = Role("my_role", using="default")
            #role.create()
            #role.grant(object = "Global", object_name = "*", privilege = "*")

            # try to create new user
            utility.create_user(str(os.getenv("MILVUS_USERNAME")), str(os.getenv("MILVUS_PASSWORD")), using='default')
            # Add new user to role
            role.add_user(str(os.getenv("MILVUS_USERNAME")))

            logging.info(f" User "+str(os.getenv("MILVUS_USERNAME"))+ " created successfully")

        except Exception as e:
            logging.warning(f" There was an error creating user " + str(os.getenv("MILVUS_USERNAME")))
            logging.warning(f" Milvus initialization FAILED. Details: {e}")
            raise
            
        try:
            # try to update password of super user root
            logging.info(f" Attempting to change default password of root user")
            utility.update_password('root', old_password = 'Milvus', new_password = str(os.getenv("MILVUS_ROOTPASSWORD")), using="default")
            logging.info(f" Password of root user changed")
                
        except Exception as e:
            logging.warning(f" Default password of user root was not changed")
            logging.warning(f" Milvus initialization FAILED. Details: {e}")
            raise

        milvus_disconnect()
        startup_message = f" Milvus initialization COMPLETED!"

    logging.info(startup_message)



# TODO: DELETE THIS, deprecated
def load_collection_into_memory(collection):


    # Loads collection into memory if not empty 
    # Return a True flag if the collection was lodaded into memory
    
    collection_name = str(os.getenv("MILVUS_COLLECTION"))

    if collection.is_empty == True : # Cannot load an empty collection into memory
        logging.warning(f" Collection {collection_name} is empty, cannot build index or load into memory")
        return False


    # Attempt to create an index if it doesnt exist
    INDEX_NAME = "_default_idx_"
    if INDEX_NAME not in utility.list_indexes(collection_name):

        index_params={
                        "metric_type":"IP", # Inner product as search metric
                        "index_type":"IVF_FLAT",
                        "params":{"nlist":1024}
                            }

        logging.info(f" Index not found for collection '{collection_name}'. Building index with the following params: {index_params} ")
        try:
            # drop index
            #collection.drop_index()
            print("building index for collection...")
            collection.create_index(field_name="embeddings", index_name = INDEX_NAME, index_params = index_params )
            #utility.index_building_progress("embeddings")

        except Exception as e: 
            logging.warning(f" Something went wrong when building index. INFO: {e}")
            pass
    
    try:
        # Load into memory
        logging.info(" Loading collection into memory...")
        collection.load()
        logging.info(" Collection loaded into memory!")

    except Exception as e:
        logging.warning(f" Something went wrong when loading collection to memory. INFO: {e}")
        pass

    return True
 


# Select the collection by doing the necessaty sanity checks for each level 
def select_collection(collection_name, level = 1):

    if level == 1:
        assert utility.has_collection(collection_name) == True, f"Collection '{collection_name}' does not exist. These are the collections avaliable:{utility.list_collections(timeout=None, using='default')} "
        collection = Collection(collection_name)

    if level == 2:
        assert utility.has_collection(collection_name) == True, f"Collection '{collection_name}' does not exist. These are the collections avaliable:{utility.list_collections(timeout=None, using='default')} "
        collection = Collection(collection_name)
        assert collection.is_empty == False, f"Collection '{collection_name}' is empty"

    if level ==3:
        assert utility.has_collection(collection_name) == True, f"Collection '{collection_name}' does not exist. These are the collections avaliable:{utility.list_collections(timeout=None, using='default')} "
        collection = Collection(collection_name)
        assert collection.is_empty == False, f"Collection '{collection_name}' is empty"
        assert len(utility.list_indexes(collection_name))>0, f"Collection '{collection_name}' does not have a search index"    

    logging.info(f" Collection '{collection_name}' selected")
    return collection



# Deletes previous embeddings of a list of paper_ids. Return True if successful, False if not
def delete_previous_embeddings(ids, collection):

    assert collection.is_empty == False, "Collection is empty, cannot delete previous embeddings for upsert"

    try: 
        for document_id in ids:
            # Delete all previous embeddings for a certain paper_id
            expr = f"""document_id == '{document_id}'"""
            # Get all the primary keys of the embeddigs belonging to this paper_id
            res = collection.query(expr=expr, output_fields = ["id"],)
            # Extract the pks into a lists
            primary_keys = [item['id'] for item in res]
            # Delete all the embeddigs using their primary keys
            collection.delete(expr= f"id in {primary_keys}")

    except Exception as e:
        print(f"There was an error when deleting the previous embeddings. error: {e}") 
        return False

    return True
        
    
def milvus_disconnect():
    connections.disconnect(str(os.getenv("MILVUS_ALIAS")))

def function(number):
    # Does something with the parameter
    return number + 1



# Get the MySQL database connection
def get_connection():
    conn = pymysql.connect(
        host=str(os.getenv("SQL_HOST")),
        user=str(os.getenv("SQL_USER")),
        password=str(os.getenv("SQL_PW")),
        database=str(os.getenv("SQL_DB")),
        port=int(os.getenv("SQL_PORT"))
    )
    return conn


# Gets one paper content form a single id
def get_paper_content(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""SELECT distinct content FROM papers_content WHERE document_id = '{id}'""")
    
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return result


# get all papaer contents form a list of ids
def get_all_paper_contents(ids):
    conn = get_connection()
    cursor = conn.cursor()
    # we need this if statement to prentent writing a the touple as ('id',) which will rise a syntax error
    if len(ids) == 1:
        query = f"""SELECT document_id, content FROM papers_content WHERE document_id = '{ids[0]}' """
    else:
        query = f"SELECT document_id, content FROM papers_content WHERE document_id IN {tuple(ids)}"
    cursor.execute(query)

    result = cursor.fetchall()

    cursor.close()
    conn.close()
    return result



# Takes a list of dictionaries containing the information returned by the top K
def perform_ranking(list_dict):

    all_ids = [dic["document_id"] for dic in list_dict]
    all_distances = [dic["distance"] for dic in list_dict]

    df = pd.DataFrame({"document_id" : all_ids, "distance": all_distances})

    # Grouping by 'paper_id' and calculating mean and count
    grouped_df = df.groupby('document_id').agg({'distance': ['mean', 'count']})

    # Renaming the columns for clarity
    grouped_df.columns = ['distance_mean', 'distance_count']

    # Sorting the DataFrame by 'distance' and 'count' in descending order
    sorted_df = grouped_df.sort_values(by=['distance_count','distance_mean'],
                                        ascending=[False, False])

    return sorted_df.reset_index().to_dict(orient='records')