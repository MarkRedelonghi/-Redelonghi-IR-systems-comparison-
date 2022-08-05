#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from testing.test_utils import Tester, Timer
from pathlib import Path

execTester = Tester()
queryTimer = Timer()

log = logging.getLogger("_test_")

path = str(Path(__file__).parent.parent)

corpus_amazonReviews = {
    "path" : path+"/data/corpus/amazonProductReviews.csv", 
    "format" : "CSV",
    "queries": ["I brave the mist and the fog", "Great Birthday present both for girls and boys", "cheerful", "Great Birthday present for children", "I brave the fog", "I bought this game for our 5-year-old twins",
                "Not all those who wander are lost", "error/mistake", "199", "90-100", "alarming", "antlers" ], 
}

corpus_commonCrawl = {
    "path" : path+"/data/corpus/commonCrawl.json",
    "format" : "JSON",
    "queries" : ["A polar bear chases a reindeer into the water","The new animated movie introduces the world to the mythology", "apple pie", "Solr", "A reindeer was chased into water", "The new animated film shows mythology",
                 "apple pies", "1200", "12-13", "the burlgar stole the television in the house", "space with planets and stars" ]
}

corpus_ccGigafida = {
    "path" : path+"/data/corpus/ccGigafida.json",
    "format" : "JSON",
    "queries" : ["prenovljenem Frederikovem stolpu na Ljubljanskem gradu", "preko 80.000 ladij, katerih nosilnost je presegala 100 brutoregistrskih ton (BRT)", "strah in trepet", "Po sončnih pobočjih nad Kranjsko Goro",
                 "pesnik", "pesnica", "pes", "prenovljen Frederikov stolp na Ljubljanskem gradu", "80.000 ladjic z 100 tonami", "s strahovi in trepetom", "z dvema psoma", "najvišje cene goriv", "hmeljarjenje", 
                  "Kranjska gora je sončna in ima veliko pobočji", "vakcina", "spakedrati", "1500", "7.30-12.00", "solr in milvus, sistema za informacijsko poizvedovanje", "samo žlabudrati zna", "omaloževati"]
}

def testIndex(instance, corpus, progress, use_partition = True):
    log.info("Test index started:")
    execTester.start()
    try:
        instance.indexDocuments(corpus["path"], corpus["format"], progress, use_partition)
    finally:
        execTester.stop()
        log.info("Test index ended: {}".format(execTester.info()))


def testQuery(instance, corpus, progress):
    log.info("Test query started:")
    execTester.start()
    progress.setMax(len(corpus["queries"]))
    progress.start()
    try:
        for i, query in enumerate(corpus["queries"]):
            instance.searchText(query, queryTimer)
            progress.print(i+1)
    finally:
        progress.end()
        execTester.stop()
        log.info("Test query ended: {}".format(execTester.info()))

