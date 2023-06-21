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
import json
from src.neuroscience import NeuroExtractor
load_dotenv()
from langchain.chat_models import ChatOpenAI

openai.api_key=os.getenv("OPEN_AI_API_KEY")
LLMClient = ChatOpenAI(model_name="gpt-3.5-turbo-16k",openai_api_key=openai.api_key,temperature=0)

## You api key from vendors or hugginface
#openai.api_key=os.getenv("OPEN_AI_API_KEY")
#LLMClient = OpenAI(model_name='text-davinci-003', openai_api_key=openai.api_key,temperature=0)
extractor = NeuroExtractor()

async def ui_extraction(input_file, apikey, dimension):
        file_name = input_file.name.split("/")[-1]
        results = await extractor.extraction(file_name, input_file.name, apikey, dimension)
        # Build results in the correct format for the Gradio front-end
        results = pd.DataFrame(results)
        return results


## Building the layout of the app
css = """.table-wrap.scroll-hide.svelte-8hrj8a.no-wrap {
    white-space: normal;
}
#component-7 .wrap.svelte-xwlu1w {
    min-height: var(--size-40);
}
div#component-2 h2 {
    color: var(--block-label-text-color);
    text-align: center;
    border-radius: 7px;
    text-align: center;
    margin: 0 15% 0 15%;
}
div#component-5 {
    border: 1px solid var(--border-color-primary);
    border-radius: 0 0px 10px 10px;
    padding: 20px;
}
.gradio-container.gradio-container-3-26-0.svelte-ac4rv4.app {
    max-width: 850px;
}
div#component-6 {
    min-height: 150px;
}
button#component-17 {
    color: var(--block-label-text-color);
}
.gradio-container.gradio-container-3-26-0.svelte-ac4rv4.app {
    max-width: 1100px;
}
#component-9 .wrap.svelte-xwlu1w {
    min-height: var(--size-40);
}
div#component-11 {
    height: var(--size-40);
}
div#component-9 {
    border: 1px solid grey;
    border-radius: 10px;
    padding: 3px;
    text-align: center;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Row():
            gr.Markdown("## Neuroscience study analyzer")
        
    with gr.Row():
        
        with gr.Column():
           fileinput = gr.File(label="Upload the dataset documentation"),

        with gr.Column():
             gr.Markdown(""" <h4 style=text-align:center>Instructions: </h4> 
     
        <b>  &#10549; Try the examples </b> at the bottom 

         <b> then </b>

 
         <b> &#8680; Set your API key </b> of OpenAI  
        
         <b> &#8678; Upload </b> your data paper (in PDF or TXT)

         <b> &#8681; Click in get insights  </b> in one tab!


         """)
        with gr.Column():
            apikey_elem = gr.Text(label="OpenAI API key (Not needed during review)")
    with gr.Row():
        with gr.Tab("Topic"):
            gr.Markdown("""In this dimension, you can get information regarding the annotation process of the data: Extract a description of the process and infer its type. Extract the labels and information about the annotation team, the infrastructure used to annotate the data, and the validation process applied to the labels.""")
            result_anot = gr.DataFrame(headers=["dimension","result"],type="array",label="Results of the extraction:")
            button_annotation = gr.Button("Get the annotation process insights!")
                
       
    ## Events of the apps
    button_annotation.click(ui_extraction,inputs=[fileinput[0],apikey_elem,gr.State(value="topics")],outputs=[result_anot])
   
    # Run the app
    #demo.queue(concurrency_count=5,max_size=20).launch()
    demo.launch(share=False,show_api=False)
        
