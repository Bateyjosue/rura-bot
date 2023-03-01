import json
import requests
# from langdetect import detect
from fastapi import FastAPI, Request
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import detectlanguage

import train as tr


# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import re

app = FastAPI(title="chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class translator:
  api_url = "https://translate.googleapis.com/translate_a/single"
  client = "?client=gtx&dt=t"
  dt = "&dt=t"

  # fROM English to Kinyarwanda
  def translate(text: str, target_lang: str, source_lang: str):
    '''FUnction to translate text'''
    sl = f"&sl={source_lang}"
    tl = f"&tl={target_lang}"
    r = requests.get(translator.api_url + translator.client +
                      translator.dt + sl + tl + "&q=" + text)
    return json.loads(r.text)[0][0][0]


# def process_question(text: str):

#     source_lang = detect(text)
#     resp = translator.translate(text=text, target_lang='en', source_lang=source_lang)
#     return resp, source_lang


def process_answer(text: str, source_lang):
    resp = translator.translate(
        text=text, target_lang=source_lang, source_lang='en')
    return resp

def query(question : str):
  # sentence = "do you use credit cards?"
  sentence = question

  sentence = tr.tokenize(sentence)
  X = tr.bag_of_words(sentence, tr.all_words)
  X = X.reshape(1, X.shape[0])
  X = tr.torch.from_numpy(X).to(tr.device)

  output = tr.model(X)
  _, predicted = tr.torch.max(output, dim=1)

  tag = tr.tags[predicted.item()]

  probs = tr.torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.25:
      for intent in tr.intents['intents']:
          if tag == intent["tag"]:
              return f"{tr.random.choice(intent['responses'])}"
  else:
      return f"I do not understand..."

def bot(text):
  from chatterbot import ChatBot
  from chatterbot.trainers import ListTrainer
  chatbot = ChatBot("Chatpot")

  trainer = ListTrainer(chatbot)
  trainer.train([
      "Hello",
      "Welcome, friend 🤗",
  ])
  trainer.train([
      "What is your name?",
      "I'm a gov.rw chatBot. Ready to assist your",
  ])
  trainer.train([
      "help",
      "How can I help?"
  ])

  trainer.train([
      "Hello!","Hi there! How can I help you today?"
    ])
  trainer.train(["Hi!","Hey, how's it going?" ])
  trainer.train(["Good morning!","Good morning to you too! What brings you here today?"])
  trainer.train([
    "Good afternoon!","Good afternoon! How may I assist you today?"
  ])
  trainer.train(["Good evening!","Good evening! What can I help you with?"])
  trainer.train([
    "Hey!","Hi there! How can I assist you today?"
  ])
  trainer.train([
    "Nice to meet you!","Nice to meet you too! What can I help you with today?"
  ])
  trainer.train([
    "Howdy!","Howdy! What can I help you with today?"
  ]),
  trainer.train([
    "Yo!","Yo! What's up? How can I help you today?"
  ]),
  trainer.train([
    "Welcome!","Thank you! How may I assist you today?"
  ])
  trainer.train([
    "Goodbye","Goodbye, have a great day!"
  ])
  trainer.train([
    "See you later","See you later, have a good one!"
  ])
  trainer.train([
    "Bye","Bye for now, take care!"
  ])
  trainer.train([
    "Take care","You too, have a great day!"
  ])
  trainer.train([
    "Hey there","Hey, what's up?"
  ])
  trainer.train(["Sup","Not much, how about you?"])
  trainer.train([
      "I do not understand...",
      "I'm sorry, I'm a bot and I was trained to perform some specific task, Please ask me about gov.rw"
  ])

  exit_conditions = (":q", "quit", "exit", "bye", " good bye")
  while True:
    query = text
    if query in exit_conditions:
        return 'It was nice to meet you'
    else:
        return f"🤖 {chatbot.get_response(query)}"


user_message = []
bot_message = []
current_lang = ''
dic = {}

welcome = [
    'Hi! I\'m your Assistant.',
    'How Can I Help you?',
    'Please Type your Request!!!'
]

def detects(text):
  """This function is all about detect diffrent languages in a sentence and return the list of languages detect"""

  API_KEY = "005a812ac87f778d20eda4770bf10445"
  detectlanguage.configuration.api_key = API_KEY
  lang = detectlanguage.simple_detect(text)
  return lang

def translates(text, target):
  source = detects(text)
  response = translator.translate(text=text, target_lang=target, source_lang=source)
  return response

@app.get('/')  # ROUTE
def index(request: Request):
    return templates.TemplateResponse("index.html", context={
        "request": request,
        'welcome': welcome,
        'dic': dic
    })

@app.get('/chat')
def chat(request: Request, message:str):
  """ENDPOIT TO CHATBOT"""

  dic.clear()
  user_message.clear()
  bot_message.clear()


  current_lang = detects(message)
  print(current_lang, message)

  if current_lang == 'en':
    welcomes = [wel for wel in welcome]
    user_message.append(message)
    response = query(message)
    bot_message.append(bot(response)) if response == "I do not understand..." else bot_message.append(response)

    for index, i in enumerate(user_message):
      dic[f'user{index}'] = i
    for index, i in enumerate(bot_message):
      dic[f'bot{index}'] = i
    return templates.TemplateResponse("index.html", context ={
        "request":request,
        'welcome': welcomes[0:-1],
        'dic': dic
        })

  else:
    user_message.append(message)
    welcomes = [translates(txt, current_lang) for txt in welcome]
    response = query(translates(message, 'en'))
    response = bot(response) if response == "I do not understand..." else response
    message = translates(response, current_lang)
    bot_message.append(message)

    for index, i in enumerate(user_message):
      dic[f'user{index}'] = i
    for index, i in enumerate(bot_message):
      dic[f'bot{index}'] = i
    return templates.TemplateResponse("index.html", context ={
        "request":request,
        'welcome': welcomes[0:-1],
        'dic': dic
        })