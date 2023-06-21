# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from src.extractor import Extractor
from src.neuroscience import NeuroExtractor
import json
import tempfile
import os

app = FastAPI()
extractor = Extractor()

@app.get("/")
async def root():
    return {"message": "API working! go to /Docs to see the Documentation"}

@app.post("/neuroscience")
async def neuro(file:UploadFile, apikey:str):
    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        ## Save uploaded file as a tempfile
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents);
        except Exception:
            raise HTTPException(status_code=500, detail='Error on uploading the file')
        finally:
            file.file.close()    

        ## Extract dimension and completenes report  
        results = await NeuroExtractor.extraction(file.filename, temp.name, apikey, "topics")
        return JSONResponse([results])

    # Exception handler
    except Exception as err:
        if type(err).__name__ == "AuthenticationError":
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(err).__name__, err.json_body['error']['code'])
            raise HTTPException(status_code=err.http_status, detail=message)
        print(f"Unexpected {err=}, {type(err)=}")
        raise HTTPException(status_code=500, detail="something goes wrong")
    finally:
        file.file.close()
        os.remove(temp.name)  # Delete temp file