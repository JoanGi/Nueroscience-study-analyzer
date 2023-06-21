import openai
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.docstore.document import Document
import pandas as pd
import os
import scipdf ## You need a Gorbid service available
import tabula ## You need to have the Java Tabula installed in the environment
from gradio import DataFrame
import asyncio
from transformers import pipeline
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI


class NeuroExtractor:

    texts = []

    def __init__(self):
        print("Initializing extractor")
        # Init classifier for the post-processing stage
        self.classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

    async def extraction(self, file_name, file_path, apikey, dimension):
        # Build the chains
        chain_incontext = self.build_chains(apikey) 
        # Prepare the data, return a list of langchain documents
        docs = self.extract_text_clean(file_name, file_path)

        #results = pd.DataFrame(columns=['Topics', 'Subtopic','Subtopic 2','Other Topic','Model', 'Model 2', 'Method', 'Method 2'])
        #results = []
        # Extract topics
        topics = self.get_topics(docs,chain_incontext)
        # Extract study model
        models = self.get_model(docs,chain_incontext)
        # Extract study methods
        methods = self.get_methods(docs,chain_incontext)

        results = {}
        for d in (topics, models, methods): results.update(d)
        print(results)
        return results
    
    def extract_text_clean(self, file_name, file_path):
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == ".pdf":
            article_dict = scipdf.parse_pdf_to_dict(file_path, soup=True,return_coordinates=False, grobid_url="https://kermitt2-grobid.hf.space") # return dictionary
            all_texts = []
            all_texts.append("Title: "+article_dict['title'])
            all_texts.append("Authors: "+ article_dict['authors'])
            all_texts.append("Abstract: "+article_dict['abstract'])
            all_texts.append(article_dict['title'])
            for section in article_dict['sections']:
             if (section['text'] != ''):
               all_texts.append(section['heading'] + ": "+ section['text'])
            
            # Return text
            return_texts = []
            for text in all_texts:
                return_texts.append(Document(page_content=text,metadata=[]))
            return return_texts


    async def prepare_data(self, file_name, file_path, chain_table, apikey):
        # Process text and get the embeddings
        vectorspath = "./vectors/"+file_name
        if not apikey:
            apikey = openai.api_key
            gr.Error("Please set your api key")
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        if os.path.isfile(vectorspath+"/index.faiss"):

            # file exists
            docsearch = FAISS.load_local(vectorspath,embeddings=embeddings)

            print("We get the embeddings from local store")
        else:
            #progress(0.40, desc="Detected new document. Splitting and generating the embeddings")
            print("We generate the embeddings using thir-party service")
            # Get extracted running text
            self.texts = self.extract_text_clean(file_name, file_path)

            # Configure the text splitter and embeddings
            #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=450, chunk_overlap=10, separators=[".", ",", " \n\n "])
            #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=450, chunk_overlap=10, separators=[" \n\n "])
            # Split, and clean
            #texts = text_splitter.split_text(text)
            for idx, text in enumerate(self.texts):
                self.texts[idx] = text.replace('\n',' ')
            print("Creating embeddings")
            # Create an index search    
            docsearch = FAISS.from_texts(self.texts, embeddings, metadatas=[{"source": i} for i in range(len(self.texts))])
            # Extract and prepare tables
        # progress(0.60, desc="Embeddings generated, parsing and transforming tables")
            if (os.path.splitext(file_name)[1] == '.pdf'):
                docsearch = await self.get_tables(docsearch,chain_table,file_path)
            
            # Save the index locally
            FAISS.save_local(docsearch, "./vectors/"+file_name)
    
        return docsearch

    def build_chains(self, apikey):
        if not apikey:
            apikey = openai.api_key
            gr.Error("Please set your api key")
        LLMClient = ChatOpenAI(model_name="gpt-4",openai_api_key=openai.api_key,temperature=0)
        #LLMClient = ChatOpenAI(model_name="gpt-3.5-turbo-16k",openai_api_key=openai.api_key,temperature=0)
        #LLMClient = OpenAI(model_name='text-davinci-003',openai_api_key=apikey,temperature=0)
        ## In-context prompt
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Question: {question}
        ###
        Context: 
        {context}
        ###
        Helpful answer:
        """
        in_context_prompt = PromptTemplate(
            input_variables=["context","question"],
            template=prompt_template,
        )
        chain_incontext = load_qa_chain(LLMClient, chain_type="stuff", prompt=in_context_prompt)

        

        return chain_incontext


    def get_topics(self, docs, incontext_prompt):
        # Init and prepare text
        report = {}
        
            
        
        
        # Get the first level of topics
        query = """Classify the provided text into one of the following topics:
                    Topics: """                 
        topics = "Behavioral neuroscience , Cellular neuroscience , Clinical neuroscience , Cognitive neuroscience , Developmental neuroscience , Molecular neuroscience , Neurophysiology, Neuroimmunity, Nervous System Tumour (Tumor; cancer), Treatment (Therapy), Others"""
       
        result1 = incontext_prompt({"input_documents": docs, "question": query+topics},return_only_outputs=True)
        classification1 = self.classifier(result1['output_text'], ["Behavioral neuroscience" , "Cellular neuroscience" , "Clinical neuroscience" , "Cognitive neuroscience" , "Developmental neuroscience" , "Molecular neuroscience" , "Neurophysiology","Neuroimmunity", "Nervous System Tumour (Tumor; cancer)", "Treatment (Therapy)"])
        report['topic'] = [classification1['labels'][0]]
        # Behavioral neuroscience
        if(classification1['labels'][0] == 'Behavioral neuroscience'):
            topics2 = "Neuroethology, Psychology(Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists), Social neuroscience, Mental Health"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Neuroethology", "Psychology(Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists)", "Social neuroscience", "Mental Health"])
            report['subtopic'] = [classification2['labels'][0]]
            # Psychology
            if(classification1['labels'][0] == 'Psychology(Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists)'):
                topics3 = "Physiological psychology (Biological psychology; Biopsychology; Psychobiology), o	Psycholinguistics (Neurolinguistics; Neurolinguistic; Neurolinguists;Psycholinguists; Psychology of language; Language), Psychophysics, Psychophysiology (Psychophysiologists), Emotional Psychology (Emotion; emotions)"
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Physiological psychology (Biological psychology; Biopsychology; Psychobiology)","Psycholinguistics (Neurolinguistics; Neurolinguistic; Neurolinguists;Psycholinguists; Psychology of language; Language)", "Psychophysics","Psychophysiology (Psychophysiologists)"," Emotional Psychology (Emotion; emotions)"])
                report['subtopic 2'] = [classification3['labels'][0]]
            # Mental Health 
            if(classification1['labels'][0] == 'Mental Health'):
                topics3 = """Stress,  Cortisol,	Suicide,   Anxiety disorder,	Panic disorder,	Obsessive-compulsive disorder,	Phobia,	Eating disorder,	Anorexia,	Bulimia,	Obesity,	Personality disorder,	Depression,	Bipolar disorder,	Post-traumatic stress disorder (PTSD),	Psychotic disorder,	Schizophrenia,	Addiction,	Substance use disorder,	Alcohol,	Cannabis,	Cocaine,	Tobacco"""
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Stress",  "Cortisol",	"Suicide",   "Anxiety disorder",	"Panic disorder",	"Obsessive-compulsive disorder",	"Phobia",	"Eating disorder",	"Anorexia",	"Bulimia",	"Obesity",	"Personality disorder",	"Depression",	"Bipolar disorder",	"Post-traumatic stress disorder (PTSD)",	"Psychotic disorder",	"Schizophrenia",	"Addiction",	"Substance use disorder",	"Alcohol",	"Cannabis",	"Cocaine",	"Tobacco"])
                report['subtopic 2'] = [classification3['labels'][0]]
        # Cellular neuroscience 
        if(classification1['labels'][0] == 'Cellular neuroscience'):
            topics2 = "Glia(Glial cells; Neuroglia), Neurons(Nerve cells; Neuronal cells)"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Glia(Glial cells; Neuroglia)", "Neurons(Nerve cells; Neuronal cells)"])
            report['subtopic'] = [classification2['labels'][0]]
            # Glia
            if(classification1['labels'][0] == 'Glia(Glial cells; Neuroglia)'):
                topics3 = "Astrocytes (Astroglia; Astroglial cells), Microglia (Microglial cells) , Oligodendrocytes (Oligodendroglia), Radial glial cells (Radial glia) , Schwann cells (Neurilemma; Neurolemmocytes)"
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Astrocytes (Astroglia; Astroglial cells)", "Microglia (Microglial cells)" , "Oligodendrocytes (Oligodendroglia)", "Radial glial cells (Radial glia)" , "Schwann cells (Neurilemma; Neurolemmocytes)"])
                report['subtopic 2'] = [classification3['labels'][0]]
            # Neurons
            if(classification1['labels'][0] == 'Neurons(Nerve cells; Neuronal cells)'):
                topics3 = """Cortical neurons, Dopaminergic neurons , GABAergic neurons , Glutamatergic neurons , Granule cells (Granule neurons), Hippocampal neurons, Interneurons , Mirror neurons , Motor neurons (Motor nerves), Neuronal synapses, Purkinje cells , Pyramidal neurons (Pyramidal cells), Sensory neurons , Serotonergic neurons """
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Cortical neurons", "Dopaminergic neurons" , "GABAergic neurons" , "Glutamatergic neurons" , "Granule cells (Granule neurons)", "Hippocampal neurons", "Interneurons" , "Mirror neurons" , "Motor neurons (Motor nerves)", "Neuronal synapses", "Purkinje cells" , "Pyramidal neurons (Pyramidal cells)", "Sensory neurons" , "Serotonergic neurons" ])
                report['subtopic 2'] = [classification3['labels'][0]]
        
        # Clinical neuroscience 
        if(classification1['labels'][0] == 'Clinical neuroscience'):
            topics2 = "Brain stimulation, Neural prosthetics (Neuroprosthetics), Neurological disorders"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Brain stimulation", "Neural prosthetics (Neuroprosthetics)", "Neurological disorders"])
            report['subtopic'] = [classification2['labels'][0]]
            # Neurological disorders
            if(classification1['labels'][0] == 'Neurological disorders'):
                topics3 = """Rare disease,	Amnesia ,	Amyotrophic lateral sclerosis (Lateral sclerosis; Lou Gehrig disease; Lou Gehrig's disease; Motor neuron disease),	Aneurysms ,	Basal ganglia disease (Basal ganglion disease; Diseases of the basal ganglia),	Brain ischemia (Cerebral ischemia),	Cerebral palsy ,	Coma ,	Demyelinating diseases (Demyelinating; Demyelination),	Dopamine deficiency ,	Epilepsy (Seizure disorder),	Hydrocephalus (Hydrocephalic),	Neurodegenerative diseases (Neurodegeneration; Neurodegenerative disorders)  ,	Dementia,	Alzheimer disease (Alzheimer's disease; Alzheimers disease),	Amyloid,	Tau ,	Ataxia ,	Huntingtons disease (Huntington's chorea; Huntington's disease; Huntingtons chorea),	Leukodystrophy ,	Adrenoleukodystrophy (Adrenoleukodystrophic),	Metachromatic leukodystrophy ,	Parkinsons disease (Parkinson's disease),	Transmissible spongiform encephalopathy (BSE; Bovine spongiform encephalopathy; Chronic wasting disease; Kuru; Mad cow disease; Prion diseases; Scrapie) ,	Creutzfeldt Jakob disease (Creutzfeldt-Jakob disease) ,	Neuromuscular diseases (Neuromuscular disorders),	Multiple sclerosis ,	Muscular dystrophy ,	Paralysis ,	Polio (Infantile paralysis; Poliomyelitis),	Rett syndrome (Cerebroatrophic hyperammonaemia; Cerebroatrophic hyperammonemia),	Tics ,	Tourette syndrome (Tourette's syndrome),	Vegetative states ,	Wallerian degeneration (Anterograde degeneration; Orthograde degeneration)"""
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Rare disease",	"Amnesia" ,	"Amyotrophic lateral sclerosis (Lateral sclerosis; Lou Gehrig disease; Lou Gehrig's disease; Motor neuron disease)","	Aneurysms ","	Basal ganglia disease (Basal ganglion disease; Diseases of the basal ganglia)","	Brain ischemia (Cerebral ischemia)","	Cerebral palsy ","	Coma ","	Demyelinating diseases (Demyelinating; Demyelination)","	Dopamine deficiency ","	Epilepsy (Seizure disorder)","	Hydrocephalus (Hydrocephalic)","	Neurodegenerative diseases (Neurodegeneration; Neurodegenerative disorders)  ","	Dementia","	Alzheimer disease (Alzheimer's disease; Alzheimers disease)","	Amyloid","	Tau ","	Ataxia ","	Huntingtons disease (Huntington's chorea; Huntington's disease; Huntingtons chorea)","	Leukodystrophy ","	Adrenoleukodystrophy (Adrenoleukodystrophic)","	Metachromatic leukodystrophy ","	Parkinsons disease (Parkinson's disease)","	Transmissible spongiform encephalopathy (BSE; Bovine spongiform encephalopathy; Chronic wasting disease; Kuru; Mad cow disease; Prion diseases; Scrapie) ","	Creutzfeldt Jakob disease (Creutzfeldt-Jakob disease) ","	Neuromuscular diseases (Neuromuscular disorders)","	Multiple sclerosis ","	Muscular dystrophy ","	Paralysis ","	Polio (Infantile paralysis; Poliomyelitis)","	Rett syndrome (Cerebroatrophic hyperammonaemia; Cerebroatrophic hyperammonemia)","	Tics ","	Tourette syndrome (Tourette's syndrome)","	Vegetative states ","	Wallerian degeneration (Anterograde degeneration; Orthograde degeneration)"])
                report['subtopic 2'] = [classification3['labels'][0]]

        # Cognitive neuroscience 
        if(classification1['labels'][0] == 'Cognitive neuroscience'):
            topics2 = "Cognition, Consciousness , Neuroeconomics (Neuroeconomists; Neuromarketing), Reaction time ,Memory"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Cognition","Consciousness","Neuroeconomics (Neuroeconomists; Neuromarketing)"," Reaction time ","Memory"])
            report['subtopic'] = [classification2['labels'][0]]
        
        # Developmental neuroscience
        if(classification1['labels'][0] == 'Developmental neuroscience'):
            topics2 = "Axon growth, Brain development, Cognitive development, motor development, steam cells, Neurogenesis"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Axon growth","Brain development"," Cognitive development"," motor development"," steam cells"," Neurogenesis"])
            report['subtopic'] = [classification2['labels'][0]]
            # •	Cognitive development 
            if(classification1['labels'][0] == 'Cognitive development '):
                topics3 = """Autism (Autism spectrum disorder),	Intellectual disability (Mental retardation) ,	Down syndrome,	Fetal alcohol syndrome,	Fragile X syndrome,	Learning ,	Learning disabilities ,	Attention deficit disorder ,	Attention deficit hyperactivity disorder (ADHD),	Dyscalculia ,	Dyslexia """
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Autism (Autism spectrum disorder)","Intellectual disability (Mental retardation)" ,"Down syndrome",	"Fetal alcohol syndrome","Fragile X syndrome","Learning","Learning disabilities","Attention deficit disorder","Attention deficit hyperactivity disorder (ADHD)","Dyscalculia ","Dyslexia"])
                report['subtopic 2'] = [classification3['labels'][0]]

        # Molecular neuroscience 
        if(classification1['labels'][0] == 'Molecular neuroscience'):
            topics2 = "Metabolism, Proteins, Genertics, Autopahgy, Apoptosis"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Metabolism", "Proteins", "Genertics", "Autopahgy", "Apoptosis"])
            report['subtopic'] = [classification2['labels'][0]]
            # 3trh level
            if(classification1['labels'][0] == 'Metabolism' or classification1['labels'][0] == 'Proteins' or classification1['labels'][0] == 'Genetics'):
                topics3 = """Mitochondrion, Receptor, Enzyme, Antibody, Antigen, DNA,  RNA, Epigenetics"""
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Mitochondrion"," Receptor"," Enzyme"," Antibody"," Antigen"," DNA","  RNA"," Epigenetics"])
                report['subtopic 2'] = [classification3['labels'][0]]
        
        # Neurophysiology
        if(classification1['labels'][0] == 'Neurophysiology'):
            topics2 = "Brain, Cerebellum, Spinal Chord, Peripehral nervous system, Injury, Motor control, Myelination, Neural mechanisms, Neural Pathways, Neuromusucal junctions, Neuroplasticity, Hormone, Neurotransmission, Sensory systems, Sleep"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Brain"," Cerebellum"," Spinal Chord"," Peripehral nervous system"," Injury"," Motor control"," Myelination"," Neural mechanisms"," Neural Pathways"," Neuromusucal junctions"," Neuroplasticity"," Hormone","Neurotransmission"," Sensory systems"," Sleep"])
            report['subtopic'] = [classification2['labels'][0]]
            if(classification1['labels'][0] == 'Brain'):
                topics3 = """Cortex, Hippocampus, Hypothalamus, Accumbens, Amygdala, Striatum, Basal Ganglia, Connectivity"""
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Cortex"," Hippocampus"," Hypothalamus"," Accumbens"," Amygdala"," Striatum"," Basal Ganglia"," Connectivity"])
                report['subtopic 2'] = [classification3['labels'][0]]
            if(classification1['labels'][0] == 'Neurotransmission'):
                topics3 = """Innvervation, Nerve impulses, Neural inhibition, Neuromodulation (Neural modulation), Neuroreceptors, Neurotransmitters (Amines (N-terminal; N-terminus), Dopamine , GABA (gamma Aminobutyric acid; gamma-Aminobutyric acid), Glutamates , Orexin (Hypocretin), Serotonin)"""
                result3 = incontext_prompt({"input_documents": docs, "question": query+topics3},return_only_outputs=True)
                classification3 = self.classifier(result3['output_text'],["Innvervation", "Nerve impulses", "Neural inhibition", "Neuromodulation (Neural modulation)", "Neuroreceptors (Amines (N-terminal; N-terminus), Dopamine , GABA (gamma Aminobutyric acid; gamma-Aminobutyric acid), Glutamates , Orexin (Hypocretin), Serotonin"])
                report['subtopic 2'] = [classification3['labels'][0]]
        
        # Neuroimmunity
        if(classification1['labels'][0] == 'Neuroimmunity'):
            topics2 = "Autoimmunity, Inflammation (Enchephalitis), Inflammation (Enchephalomyelitis), Neuroinflammation"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Autoimmunity"," Inflammation (Enchephalitis)"," Inflammation (Enchephalomyelitis)"," Neuroinflammation"])
            report['subtopic'] = [classification2['labels'][0]]
        
        # Nervous System Tumour
        if(classification1['labels'][0] == 'Nervous System Tumour (Tumor; cancer)'):
            topics2 = """Glioblastoma, Flioma, Neuroblastoma, Ependimoma, Adenoma, Prolactinoma, Chemotherapy, Radiotherapy"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Glioblastoma", "Flioma", "Neuroblastoma", "Ependimoma", "Adenoma", "Prolactinoma", "Chemotherapy", "Radiotherapy"])
            report['subtopic'] = classification2['labels'][0]
        
        # Therapy
        if(classification1['labels'][0] == 'Treatment (Therapy)'):
            topics2 = "Steam cells, Rehabilitation, Anaesthesia, Pharmacology, Transplantation, Neurosurgery"""
            result2 = incontext_prompt({"input_documents": docs, "question": query+topics2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Steam cells"," Rehabilitation"," Anaesthesia"," Pharmacology"," Transplantation"," Neurosurgery"])
            report['subtopic'] = [classification2['labels'][0]]
        
        # Other topics
         # Get the first level of topics
        query = """Which of the following topics best matches the provided study:
                    Topics: diet, sport, video-game, video game, air pollution, environment, school, artificial intelligence, music, education, phone, violence, microbiota"""
        result1 = incontext_prompt({"input_documents": docs, "question": query},return_only_outputs=True)
        classification2 = self.classifier(result1['output_text'],["diet", "sport", "video-game", "video game", "air pollution", "environment", "school", "artificial intelligence", "music", "education", "phone", "violence", "microbiota"])
        report['other topics'] = [classification2['labels'][0]]
        ## TO DO: Check the Score, and set a threshold to allow multiplice other topics

        return report

    def get_model(self, docs,incontext_prompt):
        # Init and prepare text
        report = {}
        
        # Get the first level of topics
        query = """Which of the following experimental model have been used in the study?"""
        models ="\n Experimental models: In silico (Computational model; Computational neuroscience; Computational neuroscientists; Neuroinformatics; Neuroinformaticians) model, In vitro (Culture) model, In vivo (Animal model; Transgenic), Humans models """
        result1 = incontext_prompt({"input_documents": docs, "question": query+models},return_only_outputs=True)
        classification1 = self.classifier(result1['output_text'], ["In silico (Computational model; Computational neuroscience; Computational neuroscientists; Neuroinformatics; Neuroinformaticians)"," In vitro (Culture)","In vivo (Animal model; Transgenic)","In humans" ])
        report['model'] = [classification1['labels'][0]]
        if(classification1['labels'][0] == 'In vivo (Animal model; Transgenic)'):
                models2 = "C. elegans, Drosophila, Mouse (mice), Rat, Sheep, Dog, Monkey, apes, Non-human primates"""
                result2 = incontext_prompt({"input_documents": docs, "question": query+models2},return_only_outputs=True)
                classification2 = self.classifier(result2['output_text'],["C. elegans", "Drosophila", "Mouse (mice)", "Rat", "Sheep", "Dog", "Monkey", "apes", "Non-human primates"])
                report['model 2'] = [classification2['labels'][0]]
        if(classification1['labels'][0] == 'In humans'):
                query = """The experimental model are in humans. From the following classes, which best matches the kind of human used as experimental model?"""
                models2 = "Male, Female, Embryo (pregnancy), Embryo (Prenatal), Baby, Children (kid, childhood), Adolescent, Young-adult, Adult, Elder (Geriatry, Geriatric)"""
                result2 = incontext_prompt({"input_documents": docs, "question": query+models2},return_only_outputs=True)
                classification2 = self.classifier(result2['output_text'],["Male", "Female", "Embryo (pregnancy)", "Embryo (Prenatal)", "Baby", "Children (kid, childhood)", "Adolescent", "Young-adult", "Adult", "Elder (Geriatry, Geriatric)"])
                report['model 2'] = [classification2['labels'][0]]
        return report

    def get_methods(self, docs,incontext_prompt):
        # Init and prepare text
        report = {}
          # Get the first level of topics
        query = """Which of the research methods have been used in the study?"""
        methods ="\n Histology, Cell culture, Stereotactics, Optogenetics, Molecular biology (Western Blott), Molecular biology (PCR), Bioinformatics, Imaging (Neuroimaging; Brain mapping; Brain maps; Brain scans; Neuroradiologists; Neuroradiology), Immunohistochemistry, Immunofluorescence, In situ hybridization , Electroencephalography (EEG; Electroencephalogram; Electroencephalographic; Electroencephalographs) , Gene expression """
        result1 = incontext_prompt({"input_documents": docs, "question": query+methods},return_only_outputs=True)
        classification1 = self.classifier(result1['output_text'], ["Histology"," Cell culture"," Stereotactics"," Optogenetics"," Molecular biology (Western Blott)"," Molecular biology (PCR)"," Bioinformatics","Imaging (Neuroimaging; Brain mapping; Brain maps; Brain scans; Neuroradiologists; Neuroradiology)"," Immunohistochemistry"," Immunofluorescence"," In situ hybridization "," Electroencephalography (EEG; Electroencephalogram; Electroencephalographic; Electroencephalographs) "," Gene expression """])
        report['method'] = [classification1['labels'][0]]
        if(classification1['labels'][0] == "Imaging (Neuroimaging; Brain mapping; Brain maps; Brain scans; Neuroradiologists; Neuroradiology)"):
            models2 = "Electorn Microscopy, Confocal Microscopy, Two-photon Microscopy, Magnetic resonance imaging (MRI), X-Ray, Positron Emission (PET), Single Photon Emission Tomographic bioimaging, Magnetoenchephalography, Multiphoton Brain Imaging, 3D visualization, Brain activity maps (Human brain mapping; Mapping brain activity), Calcium imaging , Connectomics (Connectomes)"
            result2 = incontext_prompt({"input_documents": docs, "question": query+models2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Electorn Microscopy", "Confocal Microscopy", "Two-photon Microscopy", "Magnetic resonance imaging (MRI)", "X-Ray", "Positron Emission (PET)", "Single Photon Emission Tomographic bioimaging", "Magnetoenchephalography", "Multiphoton Brain Imaging", "3D visualization", "Brain activity maps (Human brain mapping; Mapping brain activity)", "Calcium imaging" , "Connectomics (Connectomes)]"])
            report['method 2'] = [classification2['labels'][0]]
        if(classification1['labels'][0] == "Gene expression"):
            models2 = "Proteomics, Transcriptomics, Metabolomics, Microarrays"
            result2 = incontext_prompt({"input_documents": docs, "question": query+models2},return_only_outputs=True)
            classification2 = self.classifier(result2['output_text'],["Proteomics"," Transcriptomics"," Metabolomics"," Microarrays"])
            report['method 2'] = [classification2['labels'][0]]


        return report