#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys

from datetime import date
from testing.config_tests import *
from testing.test_utils import Progress
from solr.solr import Solr
from milvus.milvus import Milvus


MILVUS_CLIENT_PARAMETERS = {
    "milvus_port":19530,
    "milvus_host":"Localhost",
    "postgres_port":"5438",
    "postgres_host":"Localhost",
}

MILVUS_CONFIG1 = {
    "name":"MILVUS CONFIG 1",
    "model": {
        "name": "average_word_embeddings_glove.6B.300d",
        "vector_size":300,
        "normalize":True,
        "max_sequence":512
    },
    "index":{
        "name":"RHNSW_SQ",
        "build_params":{"M":4,"efConstruction":128},
        "search_params":{"ef":10,}
    },
    "metric":"IP", 
}

MILVUS_CONFIG2 = {
    "name":"MILVUS CONFIG 2",
    "model": {
        "name": "average_word_embeddings_komninos",
        "vector_size":300,
        "normalize":True,
        "max_sequence":512
    },
    "index":{
        "name":"IVF_SQ8",
        "build_params":{"nlist":1024},
        "search_params":{"nprobe":10},
    },
    "metric":"IP",
}

MILVUS_CONFIG3 = {
    "name":"MILVUS CONFIG 3",
    "model": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "vector_size":384,
        "normalize":True,
        "max_sequence":128
    },
    "index":{
        "name":"IVF_SQ8",
        "build_params":{"nlist":1024},
        "search_params":{"nprobe":10},
    },
    "metric":"L2", 
}

SOLR_CLIENT_PARAMETERS = {

    "solr_host":"Localhost",
    "solr_port":8983,
}

SOLR_CONFIG1 = {
    "name":"SOLR CONFIG 1",
    "core":"english",
}

SOLR_CONFIG2 = {
    "name":"SOLR CONFIG 2",
    "core":"default",
}

SOLR_CONFIG3 = {
    "name":"SOLR CONFIG 3",
    "core":"slovene",
}

def testSolrConfig(config, action, clear):
    print("-> initializing system")
    solr = None
    corpus = None
    if(config == 1):
        solr = Solr(SOLR_CLIENT_PARAMETERS, SOLR_CONFIG1)
        corpus = corpus_amazonReviews
    elif(config == 2):
        solr = Solr(SOLR_CLIENT_PARAMETERS, SOLR_CONFIG2)
        corpus = corpus_commonCrawl
    elif(config == 3):
        solr = Solr(SOLR_CLIENT_PARAMETERS, SOLR_CONFIG3)
        corpus = corpus_ccGigafida

    if(action == "purge"):
          solr.initCore()
          print("-> clearing data")
          solr.clear()
    else:
        if(action == "index"):
            print("-> indexing documents")
            solr.initCore()
            testIndex(solr, corpus, progress)
        elif(action == "query"):
            print("-> querying documents")
            testQuery(solr, corpus, progress)
        else:
            solr.initCore()
            print("-> indexing documents")
            testIndex(solr, corpus, progress)
            print("-> querying documents")
            testQuery(solr, corpus, progress)

        if(clear == 1):
            print("-> clearing data")
            solr.clear()

def testMilvusConfig(config, action, clear, drop):
    print("-> initializing system")
    milvus = None
    corpus = None
    if(config == 1):
        milvus =  Milvus(MILVUS_CLIENT_PARAMETERS, MILVUS_CONFIG1)
        corpus = corpus_amazonReviews
    elif(config == 2):
        milvus =  Milvus(MILVUS_CLIENT_PARAMETERS, MILVUS_CONFIG2)
        corpus = corpus_commonCrawl
    elif(config == 3):
        milvus =  Milvus(MILVUS_CLIENT_PARAMETERS, MILVUS_CONFIG3)
        corpus = corpus_ccGigafida

    try:
        if(action == "purge"):  
            print("-> clearing data")
            milvus.clear()

            print("-> dropping colections")
            milvus.drop()
        else:
            if(action == "index"):
                print("-> indexing documents")
                milvus.initServices()
                testIndex(milvus, corpus, progress)
            elif(action == "query"):
                print("-> querying documents")
                testQuery(milvus, corpus, progress)
            else:
                milvus.initServices()
                print("-> indexing documents")
                testIndex(milvus, corpus, progress)
                print("-> querying documents")
                testQuery(milvus, corpus, progress)

            if(clear == 1):
                print("-> clearing data")
                milvus.clear()


            if(drop == 1):
                print("-> dropping colections")
                milvus.drop()
            else:
                print("-> releasing collections from memory")
                milvus.release()
    finally:
        milvus.disconnect()


output_width = 50
progress = Progress(output_width)
print_format = "| {:^"+str(output_width - 4)+"} |"
border = "-"*output_width

def runTests(system, config, action, iteration, clear, drop):
    print("\n" + border + "\n" + print_format.format("TESTING STARTED") + "\n" + border)

    if(system == "solr"):
        print("\n" + border + "\n" + print_format.format("TESTING SOLR") + "\n" + border)

        for i in range(iteration):
            print("\nTest {}.".format(i+1))
            testSolrConfig(config, action, clear)
            print()

    if(system == "milvus"):
        print("\n" + border + "\n" + print_format.format("TESTING MILVUS") + "\n" + border)

        for i in range(iteration):
            print("\nTest {}.".format(i+1))
            testMilvusConfig(config, action, clear, drop)
            print()

    print("\n" + border + "\n" + print_format.format("TESTING ENDED") + "\n" + border)
    print("Results in: {}\n".format(filename))

system = None
config = None
action = None
drop = 1
clear = 1
iteration = 1
filename="logs/{}.log".format(date.today().strftime("%d-%m-%Y"))

argumentList = sys.argv[1:]
arguments_num = len(argumentList)

for i in range(arguments_num):
    currentArgument = argumentList[i]

    if(arguments_num == 1 and currentArgument in ("-h", "--Help")):
        print("Script runs tests for selected system and configuration.\n"+
            " -s  --System      System to be tested. [Milvus/Solr]\n"+
            " -t  --Test        Test case to be tested. [1-3]\n"+
            " -a  --Action      Action to be executed. If none specified index and query are selected. Query option requires already indexed data. [index/query/purge]\n"+
            " -i  --Iteration   Number of tests. Default is one, which is recommended for indexing of larger files.\n"+
            " -d  --Drop        Drop table and collections, enabled by default. This only works with Milvus system. [0/1]\n"+
            " -c  --Clear       Clear data, enabled by default.  [0/1]\n"+
            " -f  --File        Specifiy absolut path for the log file. Default directory is in App/logs/. ")

        exit()

    elif(i % 2 == 0):
        currentValue = argumentList[i+1]
        if(currentArgument in ("-s", "--System")):
            system = currentValue.lower()
        elif(currentArgument in ("-c", "--Config") and currentValue.isnumeric ):
            config = int(currentValue)
        elif(currentArgument in ("-a", "--Action")):
            action = currentValue.lower()
        elif(currentArgument in ("-i", "--Iteration") and currentValue.isnumeric):
            iteration = int(currentValue)
        elif(currentArgument in ("-d", "--Drop") and currentValue.isnumeric):
            drop = int(currentValue)
        elif(currentArgument in ("-cl", "--Clear") and currentValue.isnumeric):
            clear = int(currentValue)
        elif(currentArgument in ("-f", "--File") and currentValue.isnumeric):
            filename = currentValue
        elif(currentArgument in ("-f", "--File") and currentValue.isnumeric):
            filename = currentValue

       

if(system == None or config == None):
    print("Error system and config number must be given.\nExample: -s Solr -c 1")
else:
    logging.basicConfig(stream=open(filename, 'a', encoding='utf-8'), level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')

    runTests(system, config, action, iteration, clear,  drop)

             


