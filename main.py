from fastapi import FastAPI, Request, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import re 

app = FastAPI(title="chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

dic = {}

welcome = [
  'Hi! I\'m your Assistant.',
  'How Can I Help you?',
  'Please Type your Request!!!'
]

@app.get('/')#ROUTE
def index(request: Request):
  return templates.TemplateResponse("index.html", context = {
      "request": request,
      'welcome': welcome,
      'dic': dic
      })
