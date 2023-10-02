import scipdf ## You need a Gorbid service available
from langchain.text_splitter import  SpacyTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
import os
from bs4 import BeautifulSoup
import requests as rq
import transformers
from transformers import pipeline



class PaperPreprocess():

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        query_instruction="Represent the query for retrieval: ",
        model_kwargs = {'device': 'mps'}
    )
    def extractText(self, file_path):
        print("Extracting PDF: " + file_path)
        chunk_size = 1000
        text_splitter = SpacyTextSplitter(chunk_size=1000, chunk_overlap=300)
        article_dict = scipdf.parse_pdf_to_dict(file_path, soup=True,return_coordinates=False, grobid_url="https://kermitt2-grobid.hf.space") # return dictionary
        print("PDF parsed")
        if (article_dict is not None):
            finaltext = []
            #finaltext.append("Title:"+article_dict['title']+" \n\n Authors: " + article_dict['authors'])
            finaltext.append("Title: "+article_dict['title'] +"\n\n")
            finaltext.append("Abstract: " + article_dict['abstract'])
            for section in article_dict['sections']:
                sectitle = section['heading'] + ": "
                if(isinstance(section['text'], str)):
                    res = len(section['text'].split())
                    if(res*1.33 > chunk_size):
                        #Split text
                        splittedSections = text_splitter.split_text(section['text'])
                        prevsplit = ''
                        for split in splittedSections:
                            finaltext.append( sectitle + prevsplit + split)
                            # We are loading the last sentence and appending them to the next split
                            anotherSplitter = SpacyTextSplitter(chunk_size=50, chunk_overlap=1)
                            sentences = anotherSplitter.split_text(split)
                            prevsplit = sentences[len(sentences)-1] +". "
                    else:
                        finaltext.append(sectitle + section['text']) 
                else:
                    for text in section['text']:
                        sec = sec + text+ " \n\n " 
                    res = len(sec.split())
                    if(res*1.33 > chunk_size):
                        #Split text
                        splittedSections = text_splitter.split_text(section['text'])
                        prevsplit = ''
                        for split in splittedSections:
                            finaltext.append( sectitle + prevsplit + split)
                            sentences = text_splitter.split_text(split)
                            prevsplit = sentences[len(sentences)-2] +". "+ sentences[len(sentences)-1] + ". "
                    else:
                        finaltext.append(section['heading'] +": "+sec)
                    
                    # clean \n characters
                    #for idx, text in enumerate(finaltext):
                    #    finaltext[idx] = text.replace('\n',' ')

            figures = ''
            for figure in article_dict['figures']:
                if (figure['figure_type'] == 'table'):
                    figures = figures + "\n\n In table " + figure['figure_label'] +' of the document we can see: '+ figure['figure_caption'] + " \n\n "
                else:
                    figures = figures + "\n\n In figure " + figure['figure_label'] +' of the document we can see: '+ figure['figure_caption'] + " \n\n "

            finaltext.append(figures)

            ## Check if ACK section is correctly loaded
            ##ack = False
            ##for section in article_dict['sections']:
            ##    if (section['heading'] == 'Acknowledgements'):
            ##        ack = True
            ##if ack == False:
                ##acks = get_acks(data_paper)
                ##if acks != None:
                ##    finaltext.append('Acknowledgements: '+acks)
            print("PDF parsed")
            return finaltext
        return 'error'

    def get_acks(data_paper):

        #headers = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1'}
        page = rq.get(data_paper['primary_location.landing_page_url'], allow_redirects=True)
        soup = BeautifulSoup(page.content, "html.parser")
        ack = soup.find("div", {"id": "Ack1-content"})

        if (ack == None):
            return " "
        else:
            return ack.string

    def creatEmbeddings(self, finaltext, id):
    
        if (finaltext != 'error'):
            if os.path.isfile("./vectors/"+id+"/index.faiss"):
                print("Loading embeddings")
                # Load the index
                docsearch = FAISS.load_local("./vectors/"+id,embeddings=self.embeddings)
            else:
                print("Creating embeddings")
                # Create an index search    
                docsearch = FAISS.from_texts(finaltext, self.embeddings, metadatas=[{"source": i} for i in range(len(finaltext))])
                # Save the index locally
                FAISS.save_local(docsearch, "./vectors/"+id)
            return docsearch, finaltext
        return 'error','error'