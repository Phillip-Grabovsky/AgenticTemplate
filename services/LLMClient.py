from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import json
import logging
import google.generativeai as genai
from groq import Groq

logging.basicConfig(level=logging.INFO)

class LLMClient:
    prompts = {}
    clients = {}
    models = {}
    keychain = {}
    convos = {}

    def __init__(self, prompts="services/prompts.json"):
        #Get API Keys from .env
        logging.info("Loading API keys from .env")
        load_dotenv()
        self.keychain["pplx"] = os.getenv("PPLX_API_KEY")   #Load keys from .env
        self.keychain["openAI"] = os.getenv("OPENAI_API_KEY")  
        self.keychain["nvidia"] = os.getenv("NVIDIA_API_KEY")
        self.keychain["google"] = os.getenv("GOOGLE_API_KEY")
        self.keychain["groq"] = os.getenv("GROQ_API_KEY")
        self.keychain["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
        
        #load prompts and model info dictionaries
        with open(prompts, 'r') as file:
            self.prompts = json.load(file)
        with open("services/models.json", 'r') as file:
            self.models = json.load(file)
        logging.info("LLMClient initialized")

    def clearConvos(self):
        self.convos = {}

    # return a conversation
    async def getConvo(self, convoId):
        try:
            return self.convos[convoId]
        except KeyError:
            return [{"error": "no convo exists with that name"}]

    # request an LLM to continue or start a saved conversation
    async def conversation(self, convoId, message, model, sysPrompt="", data=""):
        # setup model
        modelInfo=self.models[model]
        clientId=modelInfo[1]
        modelName=modelInfo[0]

        if clientId not in self.clients:
            baseurl=modelInfo[2]
            if clientId == "google":
                genai.configure(api_key=self.keychain[clientId])
                self.clients[clientId] = genai.GenerativeModel(modelName)
                print("Initialized Google Gemini client.")
            elif clientId == "groq":
                self.clients[clientId] = Groq(api_key=self.keychain[clientId])
                print("Initialized Groq client.")
            elif baseurl == "": #corresponding to an openAI GPT model
                client = AsyncOpenAI(api_key = self.keychain[clientId])
                self.clients[clientId] = client
            else:                #all other models including Anthropic
                client = AsyncOpenAI(
                    base_url = baseurl,
                    api_key = self.keychain[clientId]
                )
                self.clients[clientId] = client
            print("Initialized " + clientId + " client.")
            
        # setup prompts & make payload
        SP = ""
        UP = ""
        try:    #attempt to index json
            UP = self.prompts[message]
        except KeyError:
            UP = message
        try:    #attempt to index json
            SP = self.prompts[sysPrompt]
        except KeyError:
            SP = sysPrompt

        messages = await self.createPayload(sysPrompt=SP,usrPrompt=UP,convoId=convoId,data=data)

        # For Google models, handle chat differently
        if modelInfo[1] == "google":
            try:
                # Get existing chat history and convert to Gemini format
                history = []
                for msg in self.convos[convoId]:
                    if msg["role"] == "user":
                        history.append({"role": "user", "parts": [msg["content"]]})
                    elif msg["role"] == "assistant":
                        history.append({"role": "model", "parts": [msg["content"]]})
            except KeyError:
                history = []
                if SP:  # Add system prompt as first user message if provided
                    history.append({"role": "user", "parts": [SP]})
            
            chat = self.clients[clientId].start_chat(history=history)
            response = chat.send_message(UP)
            msg = response.text
        elif modelInfo[1] == "groq":
            response = self.clients[clientId].chat.completions.create(
                model=modelName,
                messages=messages
            )
            msg = response.choices[0].message.content
        else:  # OpenAI and Anthropic models
            response = await self.clients[clientId].chat.completions.create(
                model=modelName,
                messages=messages
            )
            msg = response.choices[0].message.content

        messages.append({
            "role": "assistant",
            "content": ( msg ),
        })
        self.convos[convoId] = messages
        return msg

    # request an LLM a single time, do not remember convo history
    async def oneShot(self, sysPrompt, usrPrompt, model, data=""):
        # setup model
        modelInfo=self.models[model]
        clientId=modelInfo[1]
        modelName=modelInfo[0]
        if clientId not in self.clients:
            baseurl=modelInfo[2]
            if clientId == "google":
                genai.configure(api_key=self.keychain[clientId])
                self.clients[clientId] = genai.GenerativeModel(modelName)
                print("Initialized Google Gemini client.")
            elif clientId == "groq":
                self.clients[clientId] = Groq(api_key=self.keychain[clientId])
                print("Initialized Groq client.")
            elif baseurl == "": #corresponding to an openAI GPT model
                client = AsyncOpenAI(api_key = self.keychain[clientId])
                self.clients[clientId] = client
            else:                #all other models including Anthropic
                client = AsyncOpenAI(
                    base_url = baseurl,
                    api_key = self.keychain[clientId]
                )
                self.clients[clientId] = client
            print("Initialized " + clientId + " client.")
        # setup prompts & make payload
        SP = ""
        UP = ""
        try:    #attempt to index json
            UP = self.prompts[usrPrompt]
        except KeyError:
            UP = usrPrompt
        try:    #attempt to index json
            SP = self.prompts[sysPrompt]
        except KeyError:
            SP = sysPrompt
        messages = await self.createPayload(sysPrompt=SP,usrPrompt=UP,data=data)
        # request
        if modelInfo[1] == "google":
            response = self.clients[clientId].generate_content(usrPrompt)
            msg = response.text
        elif modelInfo[1] == "groq":
            response = self.clients[clientId].chat.completions.create(
                model=modelName,
                messages=messages
            )
            msg = response.choices[0].message.content
        else:  # OpenAI and Anthropic models
            response = await self.clients[clientId].chat.completions.create(
                model=modelName,
                messages=messages,
            )
            msg = response.choices[0].message.content
        return msg

    # form messages array payload by appending new prompting & data to (optional) convo msg history
    async def createPayload(self, sysPrompt="", usrPrompt="", convoId="",data=""):
        messages = []
        try: #continue a conversation
            messages = self.convos[convoId]
            messages.append({
                    "role": "user",
                    "content": ( usrPrompt + data ),
                })
        except KeyError: #new conversation or oneshot
            messages = [
                {
                    "role": "system",
                    "content": (sysPrompt),
                },
                {
                    "role": "user",
                    "content": ( usrPrompt + data ),
                },
            ]
        finally:
            return messages