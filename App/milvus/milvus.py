#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import normalize
import psycopg2
import sys
import os 
import re
import json

from numba import jit, cuda

LOG = logging.getLogger("_milvus_")

MODELS_PATH = os.path.dirname(os.path.realpath(__file__)) + "/models/"

TABLE_NAME = "texts"
TITLE_COLLECTION = "title_collection"
TEXT_COLLECTION = "text_collection"
EMBEDDING_FIELD = "embedding_field"
PRIMARY_FIELD = "id"
TITLE_FIELD = "title_id"
PRIMARY_COLUMN = "id"
TITLE_COLUMN = "title"
TEXT_COLUMN = "text"
INSERT_TRY = 3
TITLE_PARTITION_PROCENT = 0.1 # Partition size 10%
SENTENCE_INSERT_LIMIT = 50000 # Sentence partition size for encoding

class Milvus:
    def __init__(self, client, config):
        self.log = self._get_log()
        self.host = client["milvus_host"]
        self.port = client["milvus_port"]
        self.dbhost = client["postgres_host"]
        self.dbport = client["postgres_port"]
        self.metric_type = config["metric"]
        self._setIndex(config["index"])
        if("model" in config):
            self.title_model_config = config["model"]
            self.text_model_config = config["model"]
        else:
            self.title_model_config = config["title_model"]
            self.text_model_config = config["text_model"]
        self.title_model = self._getModel(self.title_model_config["name"])
        self.text_model = self._getModel(self.text_model_config["name"])


        self.text_collection = None
        self.title_collection = None

        self._initClients()

    def _get_log(self):
        return LOG
    
    # Set index parameters
    def _setIndex(self, index):
        self.index_type = index["name"]
        self.index_build_params = index["build_params"]
        self.index_search_params = index["search_params"]

    # Initialize Milvus and Postgres clients 
    def _initClients(self):
        self.log.info("    Initializing Milvus and Postgres clients {} {} {} {} ".format(self.host, self.port, self.dbhost, self.dbport))
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.dbconn = psycopg2.connect(host=self.dbhost, port=self.dbport, user='postgres', password='postgres')
            self.dbcursor = self.dbconn.cursor()
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Initialize data base table
    def _initDatabase(self):
        self.log.info("    Initializing postgres database")
        sql = "CREATE TABLE if not exists " + TABLE_NAME + " ("+ PRIMARY_COLUMN +" bigint, "+ TITLE_COLUMN +" text, "+ TEXT_COLUMN +" text);"
        try:
            self.dbcursor.execute(sql)
            self.dbconn.commit()
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Initialize vector collection with schema ind defined index parameters
    def _createCollection(self, name, dim, index_params, additional_field = None):
        if(not utility.has_collection(name)):
            self.log.info("    Initializing collection: {}".format(name))
            id_field = FieldSchema(name = PRIMARY_FIELD,  dtype=DataType.INT64, is_primary = True, auto_id = True)
            embeddings_field = FieldSchema(name=EMBEDDING_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dim)
            fields = [id_field, embeddings_field]

            if(additional_field):
                additional_field = FieldSchema(name = additional_field['name'], dtype = additional_field['dtype'])
                fields.append(additional_field)
            
            schema = CollectionSchema(fields=fields, description = "Vector collection")
            collection =  Collection(name = name, schema = schema)
            collection.create_index(field_name = EMBEDDING_FIELD, index_params = index_params)

            return collection
            
    # Initialize collections for document fields
    def _initCollections(self):
        index_params = { "metric_type":self.metric_type, "index_type":self.index_type, "params": self.index_build_params }
        self.title_collection = self._createCollection(TITLE_COLLECTION, self.title_model_config["vector_size"], index_params)

        additional_field = { "name" : TITLE_FIELD, "dtype": DataType.INT64 }
        self.text_collection = self._createCollection(TEXT_COLLECTION, self.text_model_config["vector_size"], index_params, additional_field)
       
    # Retrieve collection by name
    def _getCollection(self, name):
        try:
            return  Collection(name = name)
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Clear collection data
    def _clearCollection(self, name):
        try:
            collection = self._getCollection(name)
            if(not collection.is_empty):
                collection.load()
                ids = [result["id"] for result in  collection.query(expr = PRIMARY_FIELD + " >= 0", output_fields = [PRIMARY_FIELD]) ]
                collection.delete(PRIMARY_FIELD +" in {}".format(ids))
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Get saved model or load new 
    def _getModel(self, name):
        model_path = MODELS_PATH + name
        if(os.path.isdir(model_path)):
            self.log.info("    Retrieving saved model: {}".format(name))

            return SentenceTransformer(model_path, device='cuda')
            # model = SentenceTransformer(name) if no GPU

        # Load model and try to block download progress output
        sys.stdout = open(os.devnull, 'w')
        model = SentenceTransformer(name, device='cuda')
        # model = SentenceTransformer(name) if no GPU
        sys.stdout = sys.__stdout__

        model.save(model_path)
       
        return model

    # Encode sentences to vector space and insert vectors to collection
    def _insertSentenceEmbeddings(self, model, do_normalize,collection, data_encode,  additional_data = None):
        self.log.info("    Embedding {}".format(len(data_encode)))
        sentence_embeddings = model.encode(sentences=data_encode, show_progress_bar=False)

        if(do_normalize):
            sentence_embeddings = normalize(sentence_embeddings)
    
        self.log.info("    Indexing")
        retry = 0
        # Retry insert if Milvus server exception occurs
        while retry < INSERT_TRY:
            try:
                # Check if title ids need to be added to collection
                if(additional_data):
                    if(len(sentence_embeddings) != len(additional_data)):
                        self.log.info("    Inserting titles ids and sentences different size {} {}". format(len(additional_data),len(sentence_embeddings) ))

                        return None 

                    self.log.info("    Inserting titles ids and sentences")

                    return collection.insert([list(sentence_embeddings), additional_data])
    
                self.log.info("    Inserting titles")

                return collection.insert([list(sentence_embeddings)])

            except Exception as ex:
                self.log.exception("    " + str(ex) )
                retry += 1
    
    # Try to find sentences with pattern, then split them in chunks in size of sequence limit of a model
    def _tokenizeText(self, text_data, sequenceLimit, pattern):
        sentences = pattern.findall(text_data)

        split_sentences = []
        for sentence in sentences:
            chunks = [sentence[i:i+sequenceLimit] for i in range(0, len(sentence), sequenceLimit)]
            split_sentences += chunks

        return split_sentences
    
    # Insert data 
    def _indexData(self, title_data, text_data, progress, use_partition):
        title_count = len(title_data)
        progress.setMax(title_count)
        progress.start()


        if(use_partition):
            row_values = [] 
            title_ids = []
            tokenized_texts = []
            sentence_pattern = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)

            # Encode all titles to vectors at once
            self.log.info("    Inserting title sentence embeddings")
            # Retrieved object contains ids, which are used for mappig sentences to titles
            title_vector_collection = self._insertSentenceEmbeddings(self.title_model, self.title_model_config["normalize"], self.title_collection, title_data)
            self.title_collection.release()

            # Calculate partition size by procentage and number of documents
            partition_size = int(title_count * TITLE_PARTITION_PROCENT)
            if(partition_size < 1 ):
                partition_size = 1
            self.log.info("    Partitioning titles {} by {} with partition size {}".format(title_count, TITLE_PARTITION_PROCENT, partition_size))

            # Partiton titles by procent to insert to db
            for i in range(len(title_vector_collection.primary_keys)):
                # Construct a data base row
                row_values.append((title_vector_collection.primary_keys[i], title_data[i], text_data[i]))

                # Get tokenized text by finding sentences and spliting them to chunks for the model
                tokenized_text = self._tokenizeText(text_data[i], self.text_model_config["max_sequence"], sentence_pattern)
                tokenized_texts += tokenized_text
                title_ids += [title_vector_collection.primary_keys[i]] * len(tokenized_text)

                # Check if partition needs to be done
                if((i > 0 and i % partition_size == 0 ) or i == title_count-1):
                    self.log.info("    Do title partition: {}/{}".format(i+1, title_count))
                    partitioned_tokenized_texts = []
                    partitioned_title_ids = []
                    # Partition sentences by insert limit
                    for p in range(len(tokenized_texts)):
                        partitioned_tokenized_texts.append(tokenized_texts[p])
                        partitioned_title_ids.append(title_ids[p])

                        # Check if partition needs to be done
                        if((p > 0 and p % SENTENCE_INSERT_LIMIT == 0) or p == len(tokenized_texts)-1):
                            self.log.info("    Do sentence partition  {}/{} because of limit {} ".format(p, len(tokenized_texts), SENTENCE_INSERT_LIMIT))
                            self._insertSentenceEmbeddings(self.text_model, self.text_model_config["normalize"], self.text_collection, partitioned_tokenized_texts, partitioned_title_ids)  

                            partitioned_tokenized_texts = []
                            partitioned_title_ids = []

                    # Insert partition to data base
                    try:
                        self.log.info("    Inserting to db")                
                        args = ','.join(self.dbcursor.mogrify("(%s,%s,%s)", i).decode('utf-8') for i in row_values)
                        sql = "INSERT INTO " + TABLE_NAME + " VALUES " + (args)
                        self.dbcursor.execute(sql)
                        self.dbconn.commit()
                        progress.print(i+1)
                    except Exception as ex:
                        self.log.exception("    Exception when inserting to db" + str(ex) )
                    finally:
                        tokenized_texts = []
                        title_ids = []
                        row_values = []
        
        else:
            # Encode all titles to vectors at once
            self.log.info("    Inserting title sentence embeddings")
            # Retrieved object contains ids, which are used for mappig sentences to titles
            title_vector_collection = self._insertSentenceEmbeddings(self.title_model, self.title_model_config["normalize"], self.title_collection, title_data)

            for i in range(len(title_vector_collection.primary_keys)):
                row_values.append((title_vector_collection.primary_keys[i], title_data[i], text_data[i]))

                sentences = sentence_pattern.findall(text_data[i])
                tokenized_texts += sentences
                title_ids += [title_vector_collection.primary_keys[i] * len(sentences)]

            self.log.info("    Inserting text sentence embeddings")
            self._insertSentenceEmbeddings(self.text_model, self.text_model_config["normalize"], self.text_collection, tokenized_texts, title_ids)  

            try:
                self.log.info("    Inserting to db")
                args = ','.join(self.dbcursor.mogrify("(%s,%s,%s)", i).decode('utf-8') for i in row_values)
                sql = "INSERT INTO " + TABLE_NAME + " VALUES " + (args)
                self.dbcursor.execute(sql)
                self.dbconn.commit()
                progress.print(title_count)
            except Exception as ex:
                self.log.exception("    Exception when inserting to db" + str(ex) )
        
        progress.end()
        
    # Initialize data base table and vector collection
    def initServices(self):
        self._initCollections()
        self._initDatabase()     

    # Read and index documents from corpus
    def indexDocuments(self, file, format, progress,  use_partition = False ): 
        self.log.info("    Indexing documents: {} {}".format(file, format))
        title_data = []
        text_data = []

        if(self.title_collection == None):
            self.title_collection = self._getCollection(TITLE_COLLECTION) 
        if(self.text_collection == None):
            self.text_collection = self._getCollection(TEXT_COLLECTION) 
        
        if(format == "CSV" ):
            data_not_filtered = pd.read_csv(file, keep_default_na=False)
            data = data_not_filtered[['title', 'text']]
            title_data = data['title'].tolist()
            text_data = data['text'].tolist()

            self._indexData(title_data, text_data, progress, use_partition)
            
        elif(format == "JSON"):
            with open(file, 'r', encoding="utf-8") as json_file:
                docs = json.load(json_file)
                for doc in docs:
                    title_data.append(doc)
                    text_data.append(docs[doc])
                
                self._indexData(title_data, text_data, progress, use_partition)

    # Search title field
    def search(self, query):
        self.log.info("    Searching: {}".format(query))

        if(self.title_collection == None):
            self.title_collection =  self._getCollection(TITLE_COLLECTION)

        if(not self.title_collection.is_empty):
            search_params = {"metric_type": self.metric_type, "params": self.index_search_params} 
            query_embeddings = []
            if(self.title_model == None):
                self.title_model = self._getModel(self.title_model_config["name"])
            embed = self.title_model.encode(sentences=query, show_progress_bar=False)
            embed = embed.reshape(1,-1)
            if(self.title_model_config["normalize"]):
                embed = normalize(embed)
            query_embeddings = embed.tolist()

            self.title_collection.load()
            results = self.title_collection.search(query_embeddings, EMBEDDING_FIELD, param=search_params, limit=9, expr=None)

            similar_titles = []
            for result in results[0]:
                sql = "select "+ TITLE_COLUMN +" from " + TABLE_NAME + " where "+ PRIMARY_COLUMN +" = " + str(result.id) + ";"
                self.dbcursor.execute(sql)
                rows=self.dbcursor.fetchall()
                self.log.info("    Rows {}".format(rows))
                if len(rows):
                    similar_titles.append((rows[0][0], result.distance))

                if(self.metric_type == "IP"):
                    similar_titles.sort(key=lambda tup: tup[1], reverse = True) 
                if(self.metric_type == "L2"):
                    similar_titles.sort(key=lambda tup: tup[1]) 
    
            self.log.info("    Search results: {}".format(str(similar_titles)))
    
    # Search text field
    def searchText(self, query, timer = None):
        if(self.text_model):

            if(utility.has_collection(TEXT_COLLECTION)):

                if(self.text_collection == None):
                    self.text_collection = self._getCollection(TEXT_COLLECTION)

                self.log.info("    Loading collection to memory")
                self.text_collection.load()

                if(not self.text_collection.is_empty):    
                    search_params = {"metric_type": self.metric_type, "params": self.index_search_params}
                    query_embeddings = []
                    if(self.text_model == None):
                        self.text_model = self._getModel(self.text_model_config["name"])
                    self.log.info("    Encoding query")
                    embed = self.text_model.encode(sentences=query,  show_progress_bar=False)
                    embed = embed.reshape(1,-1)
                    if(self.text_model_config["normalize"]):
                        embed = normalize(embed)
                    query_embeddings = embed.tolist()

                    # Mesure query elapsed time
                    if(timer != None):
                        timer.start()
                    self.log.info("    Searching text: {}".format(query))

                    # Results of vector similarity search on text collection 
                    sentence_results = self.text_collection.search(query_embeddings, EMBEDDING_FIELD, param=search_params, limit=9, expr=None)[0]
                    sentence_result_ids = [result.id for result in sentence_results]

                    # Results of title ids query on text collection
                    expr = PRIMARY_FIELD + " in {}".format(sentence_result_ids)
                    title_results = self.text_collection.query(expr = expr, output_fields = [TITLE_FIELD])
                    
                    # Map titles to score, multiple sentences with different scores can belog to the same title
                    title_scores = {}
                    for sentence_result in sentence_results:
                        for title_result in title_results:
                            if(title_result['id'] == sentence_result.id):
                                title_id = title_result[TITLE_FIELD]
                                if(not title_id in title_scores or title_scores[title_id] > sentence_result.distance):
                                    title_scores[title_id] = sentence_result.distance
                                    break
                            
                    # Postgres query, title vector ids mapped to titles and their text
                    if(len(title_scores) > 0):
                        sql = "select * from " + TABLE_NAME + " where "+ PRIMARY_COLUMN +" in (" + ",".join([str(key) for key in title_scores.keys()]) + ") ;"
                        self.dbcursor.execute(sql)
                        rows=self.dbcursor.fetchall()

                        similar_titles = []
                        for row in rows:
                            id, title, _ = row
                            score = title_scores[id]
                            similar_titles.append((id, title, score))

                        # Order ranked titles depending on the metric type
                        if(self.metric_type == "IP"):
                            similar_titles.sort(key=lambda tup: tup[2], reverse = True) 
                        if(self.metric_type == "L2"):
                            similar_titles.sort(key=lambda tup: tup[2]) 
                        if(timer != None):
                            timer.stop()
                            self.log.info("    Query elapsed time: {}".format(timer.info()))
                        self.log.info("    Search results: {}".format(str(similar_titles)))
                    else:
                        self.log.info("    No titles found with title results: {}".format(title_results))

    # Clear table and colllection data
    def clear(self):
        self.log.info("    Milvus clearing table and collections")
        try:
            if(utility.has_collection(TITLE_COLLECTION)):
                self._clearCollection(TITLE_COLLECTION)
            if(utility.has_collection(TEXT_COLLECTION)):
                self._clearCollection(TEXT_COLLECTION)
            self.dbcursor.execute("DELETE FROM " + TABLE_NAME + " WHERE " + PRIMARY_COLUMN + " >= 0" )
            self.dbconn.commit()
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Drop table and collection data
    def drop(self):
        self.log.info("    Milvus dropping table and collections")
        try:
            if(utility.has_collection(TITLE_COLLECTION)):
                utility.drop_collection(TITLE_COLLECTION)
            if(utility.has_collection(TEXT_COLLECTION)):
                utility.drop_collection(TEXT_COLLECTION)
            self.dbcursor.execute("DROP TABLE IF EXISTS " + TABLE_NAME)
            self.dbconn.commit()
        except Exception as ex:
            self.log.exception("    " + str(ex) )
    
    # Disconnect clients
    def disconnect(self):
        self.log.info("    Milvus disconnecting")
        try:
            self.dbconn.close()
            connections.disconnect("default")
        except Exception as ex:
            self.log.exception("    " + str(ex) )
    
    # Release collections from memory
    def release(self):
        if(self.title_collection == None):
            self.title_collection = self._getCollection(TITLE_COLLECTION)
        if(self.text_collection == None):
            self.text_collection = self._getCollection(TEXT_COLLECTION)
        self.log.info("    Releasing title collection from memory")
        self.title_collection.release()
        self.log.info("    Releasing text collection from memory")
        self.text_collection.release()
