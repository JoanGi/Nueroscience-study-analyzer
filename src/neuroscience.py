import openai
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from src.extractor import PaperPreprocess
from langchain.schema import (
    HumanMessage,
    SystemMessage, 
    BaseMessage
)
from transformers import pipeline


class NeuroExtractor:

    #chat = ChatOpenAI(model="gpt-4",openai_api_key="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", temperature=0)

    def chatCall35Instruct(self, system,message):
        openai.api_key = "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"
        response = openai.Completion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": message,
                }
            ]
        )
        return response['choices'][0]['message']['content']
        
    def chatCall(self, system,message):
        openai.api_key = "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"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": message,
                }
            ]
        )
     
        return response['choices'][0]['message']['content']
    #llama = HuggingFacePipeline.from_model_id(
    #    model_id="meta-llama/Llama-2-13b-hf",
    #    task="text-generation",
    #    model_kwargs={"temperature": 0, "max_length": 256},
    #)
    #template = '''   <s>[INST] <<SYS>>
    #    {system} 
    #    <</SYS>> 
    #    {message} [/INST]'''
    #prompt = PromptTemplate(template=template, input_variables=["system","message"])
    #llamaChain = LLMChain(prompt=prompt, llm=llama)


    def LanguageModel(self, system, message):
       # messages = [
       #     SystemMessage(content=system),
       #     HumanMessage(content=message)
       # ]
        #result = self.chatCall35Instruct(system,message)
        result = self.chatCall(system,message)
        return result
       
            
        #return(HumanMessage(content="error"))


    texts = []

    def __init__(self):
        print("Initializing extractor")
        self.classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")


    async def extraction(self, file_name, file_path, apikey, dimension):
        # Build the chains
        # chain_incontext = self.build_chains(apikey) 
        # Prepare the data, return a list of langchain documents
        paperPreprocess = PaperPreprocess()
        finalTexts = paperPreprocess.extractText(file_path)
        docs = []
        for text in  finalTexts:
            docs.append(Document(page_content=text,metadata=[]))
            
        # Extract topics
        topics = self.get_topics(self.clean_intro(finalTexts))
        # Extract study model
        models = self.get_model(self.clean_text(finalTexts))
        # Extract study methods
        methods = self.get_methods(self.clean_text(finalTexts))

        results = {}
        for d in (topics, models, methods): results.update(d)
        print(results)
        return results


    def get_topics(self, docs):

        report = {} 

        ##
        # All approach
        ##
        systemAll = """Classify the provided text into the following topics:
                    Answer solely with the provided topics. If there are more than one topic, answer with the topics separated by commas.

                    Example Answer: Topic1 , Topic2 
                    Example Answer: Topic"""
        messageAll = 'Which of the provided topics corresponds to the following scientific paper: ### \n\n Topics:'+self.allTopics+'\n ### \n\n Scientific Paper text: \n\n '+ docs+' \n\n ### Topics:'
        resultAll = self.LanguageModel(system=systemAll,message=messageAll)
        report['topicsAll'] = resultAll     

        ##
        # Get each level of topic separately
        ##
        systemAbstract = """Classify the provided text into the following topics:
                    Topics: {topics}
                    
                    Answer solely with the provided topics. If there are more than one topic, answer with the topics separated by commas.

                    Example Answer: Topic1 , Topic2 
                    Example Answer: Topic"""                 
        topics = "Behavioral neuroscience, Cellular neuroscience, Clinical neuroscience, Cognitive neuroscience, Developmental neuroscience, Aging(Longevity), Aging (Nursery home),  Molecular neuroscience , Neurophysiology, Neuroimmunity, Nervous System Tumour (Tumor; cancer), Treatment (Therapy), Diagnose,  diet, sport, video game, air pollution, environment, school, artificial intelligence, music, education, phone, violence, microbiota, patient, model, none"
        system = systemAbstract.format(topics=topics)
        message = 'Which of the provided topics corresponds to the following scientific paper: ### \n\n Scientific Paper text: \n\n '+ docs+' \n\n ### Topics:'
        result1 = self.LanguageModel(system=system,message=message)
        print(result1)
 
        results = result1.split(",")
        for idx, result in enumerate(results):
            classification1 = self.classifier(result, ["Behavioral neuroscience" , "Cellular neuroscience" , "Clinical neuroscience" , "Cognitive neuroscience" , "Developmental neuroscience" , "Molecular neuroscience" , "Neurophysiology","Neuroimmunity", "Nervous System Tumour (Tumor; cancer)", "Treatment (Therapy)", "Diagnose", "diet", "sport", "video game", "air pollution", "environment", "school", "artificial intelligence", "music", "education", "phone", "violence", "microbiota", "patient", "model", "none"])
            report['topic'+str(idx)] = classification1['labels'][0]
            # Behavioral neuroscience
            if(classification1['labels'][0] == 'Behavioral neuroscience'):
                topics = "Neuroethology, Psychology(Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists), Social neuroscience, Mental Health, None"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                # classification2 = self.classifier(result2['output_text'],["Neuroethology", "Psychology(Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists)", "Social neuroscience", "Mental Health", "None"])
                report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
        
                # Psychology
                if(classification1['labels'][0] == 'Psychology(Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists)'):
                    topics = "Physiological psychology (Biological psychology; Biopsychology; Psychobiology), o	Psycholinguistics (Neurolinguistics; Neurolinguistic; Neurolinguists;Psycholinguists; Psychology of language; Language), Psychophysics, Psychophysiology (Psychophysiologists), Emotional Psychology (Emotion; emotions), None"
                    system = systemAbstract.format(topics=topics)
                    resultsubTopic = self.LanguageModel(system=system,message=message)
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
                # Mental Health 
                if(classification1['labels'][0] == 'Mental Health'):
                    topics = """Stress,  Cortisol,	Suicide,   Anxiety disorder,	Panic disorder,	Obsessive-compulsive disorder,	Phobia,	Eating disorder,	Anorexia,	Bulimia,	Obesity,	Personality disorder,	Depression,	Bipolar disorder,	Post-traumatic stress disorder (PTSD),	Psychotic disorder,	Schizophrenia,	Addiction,	Substance use disorder,	Alcohol,	Cannabis,	Cocaine,	Tobacco, None"""
                    system = systemAbstract.format(topics=topics)
                    resultsubTopic = self.LanguageModel(system=system,message=message)
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
            # Cellular neuroscience 
            if(classification1['labels'][0] == 'Cellular neuroscience'):
                topics = "Glia(Glial cells; Neuroglia), Neurons(Nerve cells; Neuronal cells)"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                classification2 = self.classifier(resultsubTopic,["Glia(Glial cells; Neuroglia)", "Neurons(Nerve cells; Neuronal cells)"])
                report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
                # Glia
                if(classification1['labels'][0] == 'Glia(Glial cells; Neuroglia)'):
                    topics = "Astrocytes (Astroglia; Astroglial cells), Microglia (Microglial cells) , Oligodendrocytes (Oligodendroglia), Radial glial cells (Radial glia) , Schwann cells (Neurilemma; Neurolemmocytes)"
                    system = systemAbstract.format(topics=topics)
                    resultsubTopic = self.LanguageModel(system=system,message=message)
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
                # Neurons
                if(classification1['labels'][0] == 'Neurons(Nerve cells; Neuronal cells)'):
                    topics = """Cortical neurons, Dopaminergic neurons , GABAergic neurons , Glutamatergic neurons , Granule cells (Granule neurons), Hippocampal neurons, Interneurons , Mirror neurons , Motor neurons (Motor nerves), Neuronal synapses, Purkinje cells , Pyramidal neurons (Pyramidal cells), Sensory neurons , Serotonergic neurons """
                    system = systemAbstract.format(topics=topics)
                    resultsubTopic = self.LanguageModel(system=system,message=message)
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
            
            # Clinical neuroscience 
            if(classification1['labels'][0] == 'Clinical neuroscience'):
                topics = "Brain stimulation, Neural prosthetics (Neuroprosthetics), Neurological disorders"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                resultsSubTopicSplited = resultsubTopic.split(",")
                for result2 in resultsSubTopicSplited:
                    classification2 = self.classifier(result2,["Brain stimulation", "Neural prosthetics (Neuroprosthetics)", "Neurological disorders"])
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " +classification2['labels'][0]
                    # Neurological disorders
                    if(classification2['labels'][0] == 'Neurological disorders'):
                        topics = """Rare disease,	Amnesia , Amyotrophic lateral sclerosis (Lateral sclerosis; Lou Gehrig disease; Lou Gehrig's disease; Motor neuron disease),	Aneurysms ,	Basal ganglia disease (Basal ganglion disease; Diseases of the basal ganglia),	Brain ischemia (Cerebral ischemia),	Cerebral palsy ,	Coma ,	Demyelinating diseases (Demyelinating; Demyelination),	Dopamine deficiency ,	Epilepsy (Seizure disorder),	Hydrocephalus (Hydrocephalic),	Neurodegenerative diseases (Neurodegeneration; Neurodegenerative disorders)  ,	Dementia, Mild Cognitive Impairment,	Alzheimer disease (Alzheimer's disease; Alzheimers disease),	Amyloid,	Tau ,	Ataxia ,	Huntingtons disease (Huntington's chorea; Huntington's disease; Huntingtons chorea),	Leukodystrophy ,	Adrenoleukodystrophy (Adrenoleukodystrophic),	Metachromatic leukodystrophy ,	Parkinsons disease (Parkinson's disease),	Transmissible spongiform encephalopathy (BSE; Bovine spongiform encephalopathy; Chronic wasting disease; Kuru; Mad cow disease; Prion diseases; Scrapie) ,	Creutzfeldt Jakob disease (Creutzfeldt-Jakob disease) ,	Neuromuscular diseases (Neuromuscular disorders),	Multiple sclerosis ,	Muscular dystrophy ,	Paralysis ,	Polio (Infantile paralysis; Poliomyelitis),	Rett syndrome (Cerebroatrophic hyperammonaemia; Cerebroatrophic hyperammonemia),	Tics ,	Tourette syndrome (Tourette's syndrome),	Vegetative states ,	Wallerian degeneration (Anterograde degeneration; Orthograde degeneration)"""
                        system = systemAbstract.format(topics=topics)
                        resultsubTopic = self.LanguageModel(system=system,message=message)
                        report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
                        

            # Cognitive neuroscience 
            if(classification1['labels'][0] == 'Cognitive neuroscience'):
                topics = "Cognition, Consciousness , Neuroeconomics (Neuroeconomists; Neuromarketing), Reaction time , Memory, Conginitive reserve"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                report['topic'+str(idx)] = report['topic'+str(idx)] + "; " + resultsubTopic
            
            # Developmental neuroscience
            if(classification1['labels'][0] == 'Developmental neuroscience'):
                topics = "Axon growth, Brain development, Cognitive development, motor development, steam cells, Neurogenesis"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                resultsSubTopicSplited = resultsubTopic.split(",")
                for result2 in resultsSubTopicSplited:
                    classification2 = self.classifier(result2,["Axon growth","Brain development","Cognitive development","motor development","steam cells","Neurogenesis"])
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " +classification2['labels'][0]
                    # •	Cognitive development 
                    if(classification2['labels'][0] == 'Cognitive development'):
                        topics = """Autism (Autism spectrum disorder),	Intellectual disability (Mental retardation) ,	Down syndrome,	Fetal alcohol syndrome,	Fragile X syndrome,	Learning ,	Learning disabilities ,	Attention deficit disorder ,	Attention deficit hyperactivity disorder (ADHD),	Dyscalculia ,	Dyslexia """
                        system = systemAbstract.format(topics=topics)
                        resultsubTopic = self.LanguageModel(system=system,message=message)
                        report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"

            # Molecular neuroscience 
            if(classification1['labels'][0] == 'Molecular neuroscience'):
                topics = "Metabolism, Proteins, Genertics, Autopahgy, Apoptosis"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                resultsSubTopicSplited = resultsubTopic.split(",")
                for result2 in resultsSubTopicSplited:
                    classification2 = self.classifier(result2,["Metabolism", "Proteins", "Genertics", "Autopahgy", "Apoptosis"])
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " +classification2['labels'][0]
                    # 3trh level
                    if(classification1['labels'][0] == 'Metabolism' or classification1['labels'][0] == 'Proteins' or classification1['labels'][0] == 'Genetics'):
                        topics = """Mitochondrion, Receptor, Enzyme, Antibody, Antigen, DNA,  RNA, Epigenetics"""
                        system = systemAbstract.format(topics=topics)
                        resultsubTopic = self.LanguageModel(system=system,message=message)
                        report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
            
            # Neurophysiology
            if(classification1['labels'][0] == 'Neurophysiology'):
                topics = "Brain, Cerebellum, Spinal Chord, Peripehral nervous system, Injury, Motor control, Myelination, Neural mechanisms, Neural Pathways, Neuromusucal junctions, Neuroplasticity, Hormone, Neurotransmission, Sensory systems, Sleep"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                resultsSubTopicSplited = resultsubTopic.split(",")
                for result2 in resultsSubTopicSplited:
                    classification2 = self.classifier(result2,["Brain"," Cerebellum","Spinal Chord","Peripehral nervous system","Injury","Motor control","Myelination","Neural mechanisms","Neural Pathways","Neuromusucal junctions","Neuroplasticity","Hormone","Neurotransmission","Sensory systems","Sleep"])
                    report['topic'+str(idx)] = report['topic'+str(idx)] + "; " +classification2['labels'][0]
                    if(classification1['labels'][0] == 'Brain'):
                        topics = """Cortex, Hippocampus, Hypothalamus, Accumbens, Amygdala, Striatum, Basal Ganglia, Connectivity"""
                        system = systemAbstract.format(topics=topics)
                        resultsubTopic = self.LanguageModel(system=system,message=message)
                        report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
                    if(classification1['labels'][0] == 'Neurotransmission'):
                        topics = """Innvervation, Nerve impulses, Neural inhibition, Neuromodulation (Neural modulation), Neuroreceptors, Neurotransmitters (Amines (N-terminal; N-terminus), Dopamine , GABA (gamma Aminobutyric acid; gamma-Aminobutyric acid), Glutamates , Orexin (Hypocretin), Serotonin)"""
                        system = systemAbstract.format(topics=topics)
                        resultsubTopic = self.LanguageModel(system=system,message=message)
                        report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
                    if(classification1['labels'][0] == 'Sensor systems'):
                        topics = """Pain, Sensory receptors (sensory cells), Vision (perception), Vision diseases (eye disorder, eye disease)"""
                        system = systemAbstract.format(topics=topics)
                        resultsubTopic = self.LanguageModel(system=system,message=message)
                        report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
            
            # Neuroimmunity
            if(classification1['labels'][0] == 'Neuroimmunity'):
                topics = "Autoimmunity, Inflammation (Enchephalitis), Inflammation (Enchephalomyelitis), Neuroinflammation"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
            
            # Nervous System Tumour
            if(classification1['labels'][0] == 'Nervous System Tumour (Tumor; cancer)'):
                topics = """Glioblastoma, Flioma, Neuroblastoma, Ependimoma, Adenoma, Prolactinoma, Chemotherapy, Radiotherapy, Pediatric Cancer, Metastasis"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
            
            # Therapy
            if(classification1['labels'][0] == 'Treatment (Therapy)'):
                topics = "Steam cells, Rehabilitation, Anaesthesia, Pharmacology, Transplantation, Neurosurgery"""
                system = systemAbstract.format(topics=topics)
                resultsubTopic = self.LanguageModel(system=system,message=message)
                report['topic'+str(idx)] = report['topic'+str(idx)] + " (" + resultsubTopic+")"
        print(report)
        return report

    def get_model(self, docs):
        # Init and prepare text
        report = {}
        systemAbstract = """ Answer with the following experimental models
        {models}
        Example Answer: model1,  model2         
        Example Answer: model1"""
        models = "Experimental models: In silico (Computational model; Computational neuroscience; Computational neuroscientists; Neuroinformatics; Neuroinformaticians) model, In vitro (Culture) model, In vivo (Animal model; Transgenic), In humans, None "
        system = systemAbstract.format(models=models)
        message = 'Which of the following experimental model have been used in the following scientific paper: ### Scientific study: \n\n '+ docs+' ### Experimental Model: :'
        result = self.LanguageModel(system=system,message=message)
        results = result.split(",")
        for idx, result in enumerate(results):
            classification1 = self.classifier(result, ["In silico (Computational model; Computational neuroscience; Computational neuroscientists; Neuroinformatics; Neuroinformaticians)"," In vitro (Culture)","In vivo (Animal model; Transgenic)","In humans","None" ])
            report['model'+str(idx)] = classification1['labels'][0]
            if(classification1['labels'][0] == 'In vivo (Animal model; Transgenic)'):
                    models = "C. elegans, Drosophila, Mouse (mice), Rat, Sheep, Dog, Monkey, apes, Non-human primates"""
                    system = systemAbstract.format(models=models)
                    resultSubModel = self.LanguageModel(system=system,message=message)
                    report['model'+str(idx)] = report['model'+str(idx)] + "; " + resultSubModel 
            if(classification1['labels'][0] == 'In humans'):
                    #query = """The experimental model are in humans. From the following classes, which best matches the kind of human used as experimental model?"""
                    models = "Male, Female, Embryo (pregnancy), Embryo (Prenatal), Baby, Children (kid, childhood), Adolescent, Young-adult, Adult, Elder (Geriatry, Geriatric)"""
                    system = systemAbstract.format(models=models)
                    resultSubModel = self.LanguageModel(system=system,message=message)
                    report['model'+str(idx)] = report['model'+str(idx)] + "; " + resultSubModel 
        print(report)
        return report

    def get_methods(self, docs):
        # Init and prepare text
        report = {}
          # Get the first level of topics
        systemAbstract = """ Answer solely with the name following research methods
        Research methods: {methods}

        Answer following the format: method 1, method 2, method 3
       """
        message = 'Which of the provided research methods have been used in the following scientific paper: ### Scientific study: \n\n '+ docs+' ### Experimental Model: :'
        methods ="\n Histology, Cell culture, Stereotactics, Optogenetics, Molecular biology (Western Blott), Molecular biology (PCR), Bioinformatics, Computational Neural Networks, Mathematics, Imaging (Neuroimaging; Brain mapping; Brain maps; Brain scans; Neuroradiologists; Neuroradiology), Immunohistochemistry, Immunofluorescence, In situ hybridization , Electroencephalography (EEG; Electroencephalogram; Electroencephalographic; Electroencephalographs) , Gene expression, Gene Edition, Xenografts, Behavioral tests (behavioral analysis; cognitive tests) """
        system = systemAbstract.format(methods=methods)
        result = self.LanguageModel(system=system,message=message)
        results = result.split(",")
        for idx, result1 in enumerate(results):
            classification1 = self.classifier(result1, ["Histology","Cell culture","Stereotactics","Optogenetics","Molecular biology (Western Blott)","Molecular biology (PCR)"," Bioinformatics","Imaging (Neuroimaging; Brain mapping; Brain maps; Brain scans; Neuroradiologists; Neuroradiology)","Immunohistochemistry","Immunofluorescence","In situ hybridization","Electroencephalography (EEG; Electroencephalogram; Electroencephalographic; Electroencephalographs) ","Gene expression", "Gene Edition", "Xenografts", "Behavioral tests (behavioral analysis; cognitive tests)"""])
            report['method'+str(idx)] = classification1['labels'][0]
            if(classification1['labels'][0] == "Imaging (Neuroimaging; Brain mapping; Brain maps; Brain scans; Neuroradiologists; Neuroradiology)"):
                methods = "Electorn Microscopy, Confocal Microscopy, Two-photon Microscopy, Magnetic resonance imaging (MRI), X-Ray, Positron Emission (PET), Single Photon Emission Tomographic bioimaging, Magnetoenchephalography, Multiphoton Brain Imaging, 3D visualization, Brain activity maps (Human brain mapping; Mapping brain activity), Calcium imaging , Connectomics (Connectomes)"
                system = systemAbstract.format(methods=methods)
                result = self.LanguageModel(system=system,message=message)
                report['method'+str(idx)] = report['method'+str(idx)] +"; " + result
            if(classification1['labels'][0] == "Gene expression"):
                methods = "Proteomics, Transcriptomics, Metabolomics, Microarrays"
                system = systemAbstract.format(methods=methods)
                result = self.LanguageModel(system=system,message=message)
                report['method'+str(idx)] = report['method'+str(idx)] +"; " + result
        print(report)
        return report

    def clean_intro(self, docs):
        # Split, and clean
        texts = ""
        count = 0
        for text in docs:
          
            if (count < 4):
                texts = texts + text.replace('\n',' ') + '''
            
                '''
            count = count + 1
        return texts
    def clean_text(self, docs):
    # Split, and clean
        texts = ""
        count = 0
        for text in docs:
            if (count < 24):
                texts = texts + text.replace('\n',' ') + '''
        
            '''
            count = count + 1
        return texts

    allTopics = """
1. Behavioral neuroscience (Behavioral neurobiologists; Behavioral neurobiology; Behavioral neuroscientists; Biological psychologists; Biological psychology; Biopsychological; Biopsychologists; Biopsychology; Psychobiological; Psychobiologists; Psychobiology)
    ●	Neuroethology 
    ●	Psychology (Affective neuroscience; Affective neuroscientists; Neuropsychological; Neuropsychologists)
        o	Physiological psychology (Biological psychology; Biopsychology; Psychobiology)
        o	Psycholinguistics (Neurolinguistics; Neurolinguistic; Neurolinguists; Psycholinguists; Psychology of language; Language)
            ▪	Bilingualism
        o	Psychophysics 
        o	Psychophysiology (Psychophysiologists)
        o	Emotional Psychology (Emotion; emotions)    
    ●	Social neuroscience (Social neuroscientists)
    ●	Mental Health
        o	Stress
            ▪	Cortisol
        o	Suicide
        o	Anxiety disorder
            ▪	Panic disorder
            ▪	Obsessive-compulsive disorder
            ▪	Phobia
        o  	Eating disorder
            ▪	Anorexia
            ▪	Bulimia
            ▪	Obesity
        o	Personality disorder
        o	Depression
        o	Bipolar disorder
        o	Post-traumatic stress disorder (PTSD)
        o	Psychotic disorder
            ▪	Schizophrenia
        o	Addiction
            ▪	Substance use disorder
            ▪	Alcohol
            ▪	Cannabis
            ▪	Cocaine
            ▪	Tobacco
2.  Cellular neuroscience (Brain cells; Cellular neurobiologists; Cellular neurobiology; Cellular neuroscientists)
    ●	Glia (Glial cells; Neuroglia)
        o	Astrocytes (Astroglia; Astroglial cells)
        o	Microglia (Microglial cells)
        o	Oligodendrocytes (Oligodendroglia)
        o	Radial glial cells (Radial glia) 
        o	Schwann cells (Neurilemma; Neurolemmocytes)
    ●	Neurons (Nerve cells; Neuronal cells)
        o	Cortical neurons 
        o	Dopaminergic neurons 
        o	GABAergic neurons 
        o	Glutamatergic neurons 
        o	Granule cells (Granule neurons)
        o	Hippocampal neurons
        o	Interneurons 
        o	Mirror neurons 
        o	Motor neurons (Motor nerves)
        o	Neuronal synapses 
        o	Purkinje cells 
        o	Pyramidal neurons (Pyramidal cells)
        o	Sensory neurons 
        o	Serotonergic neurons 
    ●	Neurovascular

3.  Clinical neuroscience (Clinical neuroscientists)
    ●	Neurology (Neurologic; Neurological; Neurological science; Neurological scientists; Neurologists; Neurology scientists)
        o	Brain stimulation 
        o	Neural prosthetics (Neuroprosthetics)
        o	Neurological disorders (Brain diseases; Brain disorders; Cerebellar diseases; Cerebellar disorders; Diseases of the brain; Diseases of the nervous system; Disorders of the brain; Disorders of the nervous system; Nervous system diseases; Nervous system disorders; Neurologic diseases; Neurologic disorders; Neurological diseases; Neuropathology; Neuropathologist; Pathologies; Disorders; Diseases)
            ▪	Rare disease
            ▪	Amnesia 
            ▪	Amyotrophic lateral sclerosis (Lateral sclerosis; Lou Gehrig disease; Lou Gehrig's disease; Motor neuron disease)
            ▪	Aneurysms 
            ▪	Basal ganglia disease (Basal ganglion disease; Diseases of the basal ganglia)
            ▪	Brain ischemia (Cerebral ischemia)
            ▪	Cerebral palsy 
            ▪	Coma 
            ▪	Demyelinating diseases (Demyelinating; Demyelination)
            ▪	Dopamine deficiency 
            ▪	Epilepsy (Seizure disorder)
            ▪	Hydrocephalus (Hydrocephalic)
            ▪	Neurodegenerative diseases (Neurodegeneration; Neurodegenerative disorders)  
            ▪	Dementia
            ▪	Mild Cognitive Impairment
            ▪	Alzheimer disease (Alzheimer's disease; Alzheimers disease)
            ▪	Amyloid
            ▪	Tau 
            ▪	Ataxia 
            ▪	Huntingtons disease (Huntington's chorea; Huntington's disease; Huntingtons chorea)
            ▪	Leukodystrophy 
            ▪	Adrenoleukodystrophy (Adrenoleukodystrophic)
            ▪	Metachromatic leukodystrophy 
            ▪	Parkinsons disease (Parkinson's disease)
            ▪	Transmissible spongiform encephalopathy (BSE; Bovine spongiform encephalopathy; Chronic wasting disease; Kuru; Mad cow disease; Prion diseases; Scrapie) 
            ▪	Creutzfeldt Jakob disease (Creutzfeldt-Jakob disease) 
            ▪	Neuromuscular diseases (Neuromuscular disorders)
            ▪	Multiple sclerosis 
            ▪	Muscular dystrophy 
            ▪	Paralysis 
            ▪	Polio (Infantile paralysis; Poliomyelitis)
            ▪	Rett syndrome (Cerebroatrophic hyperammonaemia; Cerebroatrophic hyperammonemia)
            ▪	Tics 
            ▪	Tourette syndrome (Tourette's syndrome)
            ▪	Vegetative states 
            ▪	Wallerian degeneration (Anterograde degeneration; Orthograde degeneration) 
4.  Cognitive neuroscience (Cognitive architecture; Cognitive neurobiologists; Cognitive neurobiology; Cognitive neuroscientists)
    ●	Cognition
    ●	Consciousness
    ●	Neuroeconomics (Neuroeconomists; Neuromarketing)
    ●	Reaction time
    ●	Memory
    ●	Cognitive reserve
    
5.  Developmental neuroscience (Developmental neurobiology; Developmental neuroscientists; Neural development; Neurodevelopment) 
    ●	Axon growth (Axon extension; Axon outgrowth)
    ●	Brain development (Neural development)
    ●	Cognitive development 
        o	Autism (Autism spectrum disorder)
        o	Intellectual disability (Mental retardation) 
            ▪	Down syndrome
            ▪	Fetal alcohol syndrome
            ▪	Fragile X syndrome
        o	Learning 
        o	Learning disabilities 
            ▪	Attention deficit disorder 
            ▪	Attention deficit hyperactivity disorder (ADHD)
            ▪	Dyscalculia 
            ▪	Dyslexia 
    ●	Motor development (Hand-eye coordination; Motor skill development)
    ●	Stem cells 
    ●	Neurogenesis 

6.  Aging
    ●	Longevity
    ●	Nursery home

7. Molecular neuroscience (Molecular neurobiologists; Molecular neurobiology; Molecular neuroscientists)
    ●	Metabolism
        o	Mitochondrion (Mitochondrial)
    ●	Proteins (Protein)
        o	Receptor
        o	Enzyme
        o	Antibody
        o	Antigen
    ●	Genetics (Gene)
        o	DNA
        o	RNA (Transciptome)
        o	Epigenetics
    ●	Autophagy
    ●	Apoptosis
8.  Neurophysiology (Neurophysiological; Neurophysiologists)
    ●	Brain 
        o	Cortex
        o	Hippocampus
        o	Hypothalamus
        o	Accumbens
        o	Amygdala
        o	Striatum
        o	Basal Ganglia
        o	Connectivity
    ●	Cerebellum
    ●	Spinal chord
    ●	Peripheral nervous system
        o	Nerve
    ●	Injury
        o	Traumatic brain injury
        o	Lesion
    ●	Motor control
    ●	Myelination (Myelinating; Remyelinating; Remyelination)
    ●	Neural mechanisms 
    ●	Neural pathways
    ●	Neuromuscular junctions (Neuromuscular synapses)
    ●	Neuroplasticity (Brain plasticity; Neuronal plasticity)
    ●	Hormone (Hormones)
        o	Estradiol
        o	Testosterone
    ●	Neurotransmission 
        o	Innervation 
        o	Nerve impulses 
        o	Neural inhibition 
            ▪	GABAergic inhibition (GABA reuptake inhibitors)
            ▪	Tonic inhibition 
        o	Neuromodulation (Neural modulation)
        o	Neuroreceptors (Neurotransmitter receptors)
        o	Neurotransmitters 
            ▪	Amines (N-terminal; N-terminus)
            ▪	Dopamine 
            ▪	GABA (gamma Aminobutyric acid; gamma-Aminobutyric acid)
            ▪	Glutamates 
            ▪	Orexin (Hypocretin)
            ▪	Serotonin 
    ●  	Sensory systems (Sensory circuits)
        o	Pain
        o	Sensory receptors (Sensory cells)
        o	Vision
            ▪	Visual perception
            ▪	Visual diseases (eye disorder, eye disease)
    ●	Sleep (Sleeping; Slumber)
9. Neuroimmunity
    ●	Autoimmunity (Autoimmune)
    ●	Inflammation
        o	Encephalitis
        o	Encephalomyelitis
    ●	Neuroinflammation
10. Nervous System Tumour (Tumor; cancer)
    o	Glioblastoma
    o	Glioma
    o	Neuroblastoma
    o	Ependimoma
    o	Adenoma
    o	Prolactinoma
    o	Chemotherapy
    o	Radiotherapy
    o	Pediatric cancer
    o	Metastasis
11. Treatment (Therapy)
    ●	Stem cells
    ●	Rehabilitation
    ●	Anaesthesia
    ●	Pharmacology (Drug; Medication)
    ●	Transplantation
    ●	Neurosurgery (Brain surgeons; Brain surgery; Neurosurgeons)
12. Diagnose
    ●	Marker (biomarker)
    ●	Symptom
13. Other Topics 
    ● diet 
    ● sport 
    ● video-game
    ● air pollution
    ● environment
    ● school 
    ● artificial intelligence
    ● music
    ● education
    ● phone 
    ● violence
    ● microbiota
    ● patient
    ● model  
"""