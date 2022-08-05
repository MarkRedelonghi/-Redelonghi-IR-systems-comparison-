#!/usr/bin/env python3

import json
import pandas as pd
import logging
from solr.lib.pysolr  import (Solr as SolrClient, SolrCoreAdmin)

LOG = logging.getLogger("_solr_")

LUCENE_LIMIT = 32766 # Lucene term limit of 32766 bytes
TITLE_PARTITION_PROCENT = 0.1 # Partition size 10%

class Solr:

    def __init__(self, client, config):
        self.log = self._get_log()
        self.host = client["solr_host"]
        self.port = client["solr_port"]
        self.core = config["core"]
        self.client = None

    def _get_log(self):
        return LOG

    # Initialize Solr API client for a specific core
    def _initClient(self):
        try:
            self.log.info("    Initializing Solr client with host {} port {} core {}".format(self.host, self.port, self.core))
            self.client = SolrClient("http://{}:{}/solr/{}".format(self.host, self.port, self.core))
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Search 
    def _search(self, query, timer):
        if(self.client == None):
            self._initClient()
        if(self.core == None):
            self.core = "default"
        if(timer != None):
            timer.start()

        try:
            results = self.client.search(query)
            similar_titles =  [(doc["id"], doc['title']) for doc in results.docs]

            # Mesure query elapsed time
            if(timer != None):
                timer.stop()
                self.log.info("    Query elapsed time: {}".format(timer.info()))
            self.log.info("    Search results: {}".format(similar_titles))

            return similar_titles
        except Exception as ex:
            self.log.exception("    " + str(ex) )

            return []

    def _findLastWhiteSpace(self, string, iFrom, iTo):
        i = iFrom 
        while i > iTo:
            if(string[i].isspace()):
                return i
            i-=1

        return 0
    
    # Try splitings string by white spaces and max sequence length
    def _partitionString(self,string, max):
        splits = []
        all_count = len(bytes(string.encode("utf-8")))
        byte_count = 0
        iFrom = 0   
        i = 0

        while i < len(string):
            c = string[i]
            byte_char_count = len(bytes(c.encode("utf-8")))

            if(byte_count+byte_char_count >= max):
                byte_count = 0
                last_space = self._findLastWhiteSpace(string, i, iFrom)
                if(last_space > 0):
                    splits.append((iFrom, last_space))
                    i = last_space
                    iFrom = last_space
                else:
                    splits.append((iFrom, i))
                    iFrom = i

            byte_count += byte_char_count
            i += 1
        
        splits.append((iFrom, all_count+1))

        return splits

    # Insert documents to core
    def _indexData(self, docs, progress,  use_partition):
        doc_count = len(docs)
        progress.setMax(doc_count)
        progress.start()

        if(use_partition):
            partition_docs = []
            count = 0

            partition_size = int(doc_count * TITLE_PARTITION_PROCENT)
            self.log.info("    Partitioning titles {} by {} with partition size {}".format(doc_count, TITLE_PARTITION_PROCENT, partition_size))

            for doc in docs:
                count += 1
                # used to check if last docs got indexed
                # Can't check if on last doc to index becuase the number of docs changes until end of for. Reason is spliting larger docs in multiple docs
                last_indexed = False 


                # Depending on the format docs can be a dict or array of dicts
                if(isinstance(doc, str)):
                    text = docs[doc]
                    title = doc
                else:
                    text = doc["text"]
                    title = doc["title"]

                size = len(bytes(text, 'utf-8'))

                # Check if text data size larger than lucene limit
                if(size < LUCENE_LIMIT):
                    partition_docs.append({
                        "id":count,
                        "title":title,
                        "text":text,
                    })
                else:
                    for i,split in enumerate(self._partitionString(text, LUCENE_LIMIT)):
                        last_indexed = False
                        if(i>0):
                            count += 1

                        partition_docs.append({
                            "id":count,
                            "title":title+" part "+str(i),
                            "text":text[split[0]:split[1]],
                        })
                      
                        if(count % partition_size == 0): 
                            last_indexed = True
                            self.log.info("    Adding doc partition of size {} {}/{}".format(len(partition_docs), count, doc_count ))                       
                            try:
                                self.client.add(docs=partition_docs, commit=True)
                                progress.print(count)
                            except Exception as ex:
                                self.log.exception("    " + str(ex) )
                            finally:
                                partition_docs = []

                if(count % partition_size == 0): 
                    last_indexed = True
                    self.log.info("    Adding doc partition of size {} {}/{}".format(len(partition_docs), count, doc_count))
                    try:
                        self.client.add(docs=partition_docs, commit=True)
                        progress.print(count)
                    except Exception as ex:
                        self.log.exception("    " + str(ex) )  
                    finally:
                        partition_docs = []

            # Do last partition after for if nedded
            if(not last_indexed):
                self.log.info("    Adding last docs {} {}/{}".format(len(partition_docs), count, doc_count))
                try:
                    self.client.add(docs=partition_docs, commit=True)
                    progress.print(count)
                except Exception as ex:
                    self.log.exception("    " + str(ex) )  
                finally:
                    partition_docs = []
                
        else:
            try:
                self.log.info("    Adding docs {}".format(len(docs)))
                self.client.add(docs=docs, commit=True)
                progress.start(doc_count)
            except Exception as ex:
                self.log.exception("    " + str(ex) )
            
        progress.end()

    # Initialize Solr core with CoreAdmin API
    def initCore(self):
        try:
            solr_admin = SolrCoreAdmin("http://{}:{}/solr/admin/cores".format(self.host, self.port))
            if(self.core == "default"):
                self.log.info("    Creating core with SolrCoreAdmin: http://{}:{}/solr/admin/cores {}".format(self.host, self.port, self.core))  
                response = solr_admin.create(name=self.core, instance_dir=self.core, schema="managed-schema")
            else:
                self.log.info("    Creating core with SolrCoreAdmin: http://{}:{}/solr/admin/cores {} {}".format(self.host, self.port, self.core, self.core))  
                response = solr_admin.create(name=self.core, instance_dir=self.core)

            self.log.info("    Create response: " + response)
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    # Read and index documents from corpus
    def indexDocuments(self, file, format , progress, use_partition=False):
        self.log.info("    Adding documents to index: {} {}".format(self.core, file))  
        if(self.client == None):
            self._initClient()

        if(format == "CSV"):
            data_not_filtered = pd.read_csv(file)
            # String id field
            data_not_filtered["id"] = data_not_filtered.index.astype(str)
            data = data_not_filtered[['id', 'title', 'text']]
            docs = json.loads(data.to_json(orient="records"))

            self._indexData(docs, progress, use_partition)
        elif(format == "JSON"):
            with open(file, 'r') as json_file:
                docs = json.load(json_file)

                self._indexData(docs, progress, use_partition)

    # Search title field        
    def search(self, query, timer = None):
        self.log.info('    Searching: "{}" '.format(query))
        if(self._search('title:"'+query+'"', timer) == []):
            self.log.info("    Searching: {} ".format(query))
            self._search('title:'+query, timer)
        
    # Search text field
    def searchText(self, query, timer = None):
        # try searching with and without quotation marks
        self.log.info('    Searching: "{}" '.format(query))
        self._search('text:"'+query+'"', timer)

        self.log.info("    Searching text: {} ".format(query))
        self._search('text:'+query, timer)
    
    # Clear core data
    def clear(self):
        self.log.info("    Clearing core: {}".format(self.core) )
        if(self.client == None):
            self._initClient()

        try:
            self.client.delete(q="*:*", commit=True) 
        except Exception as ex:
            self.log.exception("    " + str(ex) )

    def drop(self):
        pass