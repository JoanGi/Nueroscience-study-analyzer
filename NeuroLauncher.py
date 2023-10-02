from os import listdir
from os.path import isfile, join
from src.neuroscience import NeuroExtractor
import asyncio
import pandas as pd
import os


extractor = NeuroExtractor()
mypath = "./sources/ARTICLES/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
resultsFinal = []
resultCSV = pd.DataFrame()
for file in onlyfiles:    
    #if (file == "1.pdf" or file == "3.pdf" or file == "4.pdf" or file == "5.pdf" or file == "6.pdf" or file == "8.pdf" or file == "9.pdf" or file == "10.pdf" or file == "12.pdf"):
    results = asyncio.run(extractor.extraction(file, mypath+file, "fake", "fake"))
    results["id"] = file.split(".")[-2]
    resultsPd = pd.DataFrame([results])
    resultCSV = pd.concat([resultCSV, pd.DataFrame([resultsPd])])
    
resultCSV.to_csv("./sources/results.csv", mode='a', header=not os.path.exists("./sources/ARTICLES/results.csv"))

   
    
#resultsPd = pd.DataFrame(resultsFinal)
#resultsPd.to_csv("./sources/ARTICLES/results.csv", mode='a', header=not os.path.exists("./sources/ARTICLES/results.csv"))
       
