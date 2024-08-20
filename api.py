import re
import os
from transformers import (BartTokenizerFast, 
                          TFAutoModelForSeq2SeqLM)
import tensorflow as tf
from scraper import scrape_text
from fastapi import FastAPI, Response, Request
from typing import List
from pydantic import BaseModel, Field
from fastapi.exceptions import RequestValidationError
import uvicorn
import json
import logging
import multiprocessing


os.environ['TF_USE_LEGACY_KERAS'] = "1"
SUMM_CHECKPOINT = "facebook/bart-base"
SUMM_INPUT_N_TOKENS = 400
SUMM_TARGET_N_TOKENS = 300


def load_summarizer_models():
    summ_tokenizer = BartTokenizerFast.from_pretrained(SUMM_CHECKPOINT)
    summ_model = TFAutoModelForSeq2SeqLM.from_pretrained(SUMM_CHECKPOINT)
    summ_model.load_weights(os.path.join("models", "bart_en_summarizer.h5"), by_name=True)
    logging.warning('Loaded summarizer models')
    return summ_tokenizer, summ_model


def summ_preprocess(txt):
    txt = re.sub(r'^By \. [\w\s]+ \. ', ' ', txt) # By . Ellie Zolfagharifard . 
    txt = re.sub(r'\d{1,2}\:\d\d [a-zA-Z]{3}', ' ', txt) # 10:30 EST
    txt = re.sub(r'\d{1,2} [a-zA-Z]+ \d{4}', ' ', txt) # 10 November 1990
    txt = txt.replace('PUBLISHED:', ' ')
    txt = txt.replace('UPDATED', ' ')
    txt = re.sub(r' [\,\.\:\'\;\|] ', ' ', txt) # remove puncts with spaces before and after
    txt = txt.replace(' : ', ' ')
    txt = txt.replace('(CNN)', ' ')
    txt = txt.replace('--', ' ')
    txt = re.sub(r'^\s*[\,\.\:\'\;\|]', ' ', txt) # remove puncts at beginning of sent
    txt = re.sub(r' [\,\.\:\'\;\|] ', ' ', txt) # remove puncts with spaces before and after
    txt = re.sub(r'\n+',' ', txt)
    txt = " ".join(txt.split())
    return txt


async def summ_inference_tokenize(input_: list, n_tokens: int):
    tokenized_data = summ_tokenizer(text=input_, max_length=SUMM_TARGET_N_TOKENS, truncation=True, padding="max_length", return_tensors="tf")
    return summ_tokenizer, tokenized_data    


async def summ_inference(txts: str):
    logging.warning("Entering summ_inference()")
    txts = [*map(summ_preprocess, txts)]
    inference_tokenizer, tokenized_data = await summ_inference_tokenize(input_=txts, n_tokens=SUMM_INPUT_N_TOKENS)
    pred = summ_model.generate(**tokenized_data, max_new_tokens=SUMM_TARGET_N_TOKENS)
    result = ["" if t=="" else inference_tokenizer.decode(p, skip_special_tokens=True).strip() for t, p in zip(txts, pred)]
    return result


async def scrape_urls(urls):
    logging.warning('Entering scrape_urls()')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    results = []
    for url in urls:
        f = pool.apply_async(scrape_text, [url]) # asynchronously applying function to chunk. Each worker parallely begins to work on the job
        results.append(f) # appending result to results
        
    scraped_texts = []
    scrape_errors = []
    for f in results:
        t, e = f.get(timeout=120)
        scraped_texts.append(t)
        scrape_errors.append(e)
    pool.close()
    pool.join()
    logging.warning('Exiting scrape_urls()')
    return scraped_texts, scrape_errors


description = "API to generate summaries of news articles from their URLs."
app = FastAPI(title='News Summarizer API',
              description=description, 
              version="0.0.1",
              contact={
                  "name": "Author: KSV Muralidhar",
                  "url": "https://ksvmuralidhar.in"
              }, 
             license_info={
                 "name": "License: MIT",
                 "identifier": "MIT"
             },
             swagger_ui_parameters={"defaultModelsExpandDepth": -1})


summ_tokenizer, summ_model = load_summarizer_models()


class URLList(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles to generate summaries")
    key: str = Field(..., description="Authentication Key")

class SuccessfulResponse(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: List[str] = Field(..., description="List of scraped text from input URLs")
    scrape_errors: List[str] = Field(..., description="List of errors raised during scraping. One item for corresponding URL")
    summaries: List[str] = Field(..., description="List of generated summaries of news articles")
    summarizer_error: str = Field("", description="Empty string as the response code is 200")

class AuthenticationError(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: str = Field("", description="Empty string as authentication failed")
    scrape_errors: str = Field("", description="Empty string as authentication failed")
    summaries: str = Field("", description="Empty string as authentication failed")
    summarizer_error: str = Field("Error: Authentication error: Invalid API key.")

class SummaryError(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: List[str] = Field(..., description="List of scraped text from input URLs")
    scrape_errors: List[str] = Field(..., description="List of errors raised during scraping. One item for corresponding URL")
    summaries: str = Field("", description="Empty string as summarizer encountered an error")
    summarizer_error: str = Field("Error: Summarizer Error with a message describing the error")

class InputValidationError(BaseModel):
    urls: List[str] = Field(..., description="List of URLs of news articles inputted by the user")
    scraped_texts: str = Field("", description="Empty string as validation failed")
    scrape_errors: str = Field("", description="Empty string as validation failed")
    summaries: str = Field("", description="Empty string as validation failed")
    summarizer_error: str = Field("Validation Error with a message describing the error")


class NewsSummarizerAPIAuthenticationError(Exception):
    pass 

class NewsSummarizerAPIScrapingError(Exception):
    pass 


def authenticate_key(api_key: str):
    if api_key != os.getenv('API_KEY'):
        raise NewsSummarizerAPIAuthenticationError("Authentication error: Invalid API key.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    urls = request.query_params.getlist("urls")
    error_details = exc.errors()
    error_messages = []
    for error in error_details:
        loc = [*map(str, error['loc'])][-1]
        msg = error['msg']
        error_messages.append(f"{loc}: {msg}")
    error_message = "; ".join(error_messages) if error_messages else ""
    response_json = {'urls': urls, 'scraped_texts': '', 'scrape_errors': '', 'summaries': "", 'summarizer_error': f'Validation Error: {error_message}'}
    json_str = json.dumps(response_json, indent=5) # convert dict to JSON str
    return Response(content=json_str, media_type='application/json', status_code=422)


@app.post("/generate_summary/", tags=["Generate Summary"], response_model=List[SuccessfulResponse],
         responses={
        401: {"model": AuthenticationError, "description": "Authentication Error: Returned when the entered API key is incorrect"}, 
        500: {"model": SummaryError, "description": "Summarizer Error: Returned when the API couldn't generate the summary of even a single article"},
        422: {"model": InputValidationError, "description": "Validation Error: Returned when the payload data doesn't match the data type requirements"}
         })
async def generate_summary(q: URLList):
    """
    Get summaries of news articles by passing the list of URLs as input.

    - **urls**: List of URLs (required)
    - **key**: Authentication key (required)
    """
    try:
        logging.warning("Entering generate_summary()")
        urls = ""
        scraped_texts = ""
        scrape_errors = ""
        summaries = ""
        request_json = q.json()
        request_json = json.loads(request_json)
        urls = request_json['urls']
        api_key = request_json['key']
        _ = authenticate_key(api_key)
        scraped_texts, scrape_errors = await scrape_urls(urls)
        
        unique_scraped_texts = [*set(scraped_texts)]
        if (unique_scraped_texts[0] == "") and (len(unique_scraped_texts) == 1):
            raise NewsSummarizerAPIScrapingError("Scrape Error: Couldn't scrape text from any of the URLs")
            
        summaries = await summ_inference(scraped_texts)
        status_code = 200
        response_json = {'urls': urls, 'scraped_texts': scraped_texts, 'scrape_errors': scrape_errors, 'summaries': summaries, 'summarizer_error': ''}
    except Exception as e:
        status_code = 500
        if e.__class__.__name__ == "NewsSummarizerAPIAuthenticationError":
            status_code = 401
        response_json = {'urls': urls, 'scraped_texts': scraped_texts, 'scrape_errors': scrape_errors, 'summaries': "", 'summarizer_error': f'Error: {e}'}

    json_str = json.dumps(response_json, indent=5) # convert dict to JSON str
    return Response(content=json_str, media_type='application/json', status_code=status_code)


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=7860, workers=3)