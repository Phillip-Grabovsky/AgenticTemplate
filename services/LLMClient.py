from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import json
import logging
import google.genai as genai
from google.genai import types

logging.basicConfig(level=logging.WARNING)
# Disable AFC logging
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

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
                self.clients[clientId] = genai.Client(api_key=self.keychain[clientId])
                print("Initialized Google Gemini client.")
            elif baseurl == "": #corresponding to an openAI GPT model
                client = AsyncOpenAI(api_key = self.keychain[clientId])
                self.clients[clientId] = client
            else:                #all other openAI compatible models
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

        messages = [] #store the conversation history
        msg = ""    #store the response

        # For Google models, handle chat differently
        if modelInfo[1] == "google":
            try:
                messages, history, SP = await self.createGooglePayload(sysPrompt=SP,usrPrompt=UP,convoId=convoId,data=data)

                # Initialize new chat with system prompt if provided                    
                config = types.GenerateContentConfig(
                    system_instruction=SP
                ) if SP else None
                chat = self.clients[clientId].chats.create(
                    model=modelName,
                    history=history,
                    config=config
                )
                response = chat.send_message(UP+data)
                msg = response.text
            except Exception as e:
                logging.error(f"Error in Google chat: {str(e)}")
                raise e

        else:  # all openai compatible models
            messages = await self.createPayload(sysPrompt=SP,usrPrompt=UP,convoId=convoId,data=data)
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
                self.clients[clientId] = genai.Client(api_key=self.keychain[clientId])
                print("Initialized Google Gemini client.")
            elif baseurl == "": #corresponding to an openAI GPT model
                client = AsyncOpenAI(api_key = self.keychain[clientId])
                self.clients[clientId] = client
            else:                #all openai compatible models
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
        
        # request
        if modelInfo[1] == "google":
            config = types.GenerateContentConfig(
                system_instruction=SP
            ) if SP else None
            response = self.clients[clientId].models.generate_content(
                model=modelName,
                contents=UP+data,
                config=config
            )
            msg = response.text
        else:  # All openAI compatible models
            messages = await self.createPayload(sysPrompt=SP,usrPrompt=UP,data=data)
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


    async def createGooglePayload(self, sysPrompt="", usrPrompt="", convoId="",data=""):
        """Creates a payload for Google's Gemini API from conversation history."""
        history = []
        messages = []
        SP = sysPrompt
        
        try:
            # Get existing conversation history
            messages = self.convos[convoId]
            messages.append({
                    "role": "user",
                    "content": ( usrPrompt + data ),
                })
            
            if messages[0]["role"] == "system":
                SP = messages[0]["content"]
            
            # Convert OpenAI format messages to Google Gemini format
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                # Skip system messages as they're handled separately in the system_instruction
                if role == "system":
                    SP = content
                    continue
                    
                # Map OpenAI roles to Gemini roles
                if role == "user":
                    gemini_role = "user"
                elif role == "assistant":
                    gemini_role = "model"
                else:
                    continue  # Skip unknown roles
                
                # Create Content object for this message
                history.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part(text=content)]
                    )
                )

        except KeyError:
            # If no conversation history exists, just include the new message
            history = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=usrPrompt + data)]
                )
            ]

            # Add the new message to the conversation history
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
            return messages,history,SP
