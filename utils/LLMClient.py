from dotenv import load_dotenv
load_dotenv()

import os
import json
from openai import AsyncOpenAI
from pydantic import ValidationError
from setup_logger import setup_logger
import google.genai as genai
from google.genai import types

# Disable AFC logger
logger = setup_logger("LLMClient")

class LLMClient:

    def __init__(self):
        #Get API Keys from .env
        self.keychain = {}
        self.clients = {}
        self.prompts = {}
        self.models = {}

        self.keychain["pplx"] = os.getenv("PPLX_API_KEY")   #Load keys from .env
        self.keychain["openAI"] = os.getenv("OPENAI_API_KEY")  
        self.keychain["nvidia"] = os.getenv("NVIDIA_API_KEY")
        self.keychain["google"] = os.getenv("GOOGLE_API_KEY")
        self.keychain["groq"] = os.getenv("GROQ_API_KEY")
        self.keychain["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
        
        #load prompts and model info dictionaries
        # with open(prompts, 'r') as file:
        #     self.prompts = json.load(file)
        with open("models.json", 'r') as file:
            self.models = json.load(file)
        logger.info("LLMClient initialized")

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
            else:                #all other openai compatible models
                client = AsyncOpenAI(
                    base_url = baseurl,
                    api_key = self.keychain[clientId]
                )
                self.clients[clientId] = client
            print("Initialized " + clientId + " client.")
            
        # setup prompts & make payload
        UP = message
        SP = sysPrompt

        messages = [] #store the conversation history
        msg = ""    #store the response

        # For Google models, handle chat differently
        if modelInfo[1] == "google":
            try:
                messages, history, SP = self.createGooglePayload(sysPrompt=SP,usrPrompt=UP,convoId=convoId,data=data)

                # Initialize new chat with system prompt if provided
                print("Creating Config")               
                config = types.GenerateContentConfig(
                    system_instruction=SP
                ) if SP else None
                print("Creating Chat")
                chat = self.clients[clientId].aio.chats.create(
                    model=modelName,
                    history=history,
                    config=config
                )
                print("Sending Message")
                response = await chat.send_message(UP+data)
                print("Response Received")
                msg = response.text
            except Exception as e:
                logger.error(f"Error in Google chat: {str(e)}")
                raise e

        else:  # all openai compatible models
            messages = self.createPayload(sysPrompt=SP,usrPrompt=UP,convoId=convoId,data=data)
            response = await self.clients[clientId].chat.completions.create(
                model=modelName,
                messages=messages,
            )
            msg = response.choices[0].message.content

        messages.append({
            "role": "assistant",
            "content": ( msg ),
        })
        self.convos[convoId] = messages
        return msg

    # request an LLM a single time, do not remember convo history
    async def oneShot(self, sysPrompt: str, usrPrompt: str, model: str, data: str = "") -> str:
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
            else:                #all other openai compatible models
                client = AsyncOpenAI(
                    base_url = baseurl,
                    api_key = self.keychain[clientId]
                )
                self.clients[clientId] = client
            print("Initialized " + clientId + " client.")
        
        # Use prompts directly without trying to look them up
        SP = sysPrompt
        UP = usrPrompt
        
        # request
        if modelInfo[1] == "google":
            config = types.GenerateContentConfig(
                system_instruction=SP
            ) if SP else None
            
            # Create a chat without history
            chat = self.clients[clientId].aio.chats.create(
                model=modelName,
                config=config
            )
            response = await chat.send_message(UP+data)
            msg = response.text
        else:  # All openAI compatible models
            messages = self.createPayload(sysPrompt=SP,usrPrompt=UP,data=data)
            logger.debug(f"Messages: {str(messages)}")
            response = await self.clients[clientId].chat.completions.create(
                model=modelName,
                messages=messages
            )
            msg = response.choices[0].message.content
        return msg

    # form messages array payload by appending new prompting & data to (optional) convo msg history
    def createPayload(self, sysPrompt="", usrPrompt="", convoId="",data=""):
        messages = []
        try:
            if convoId and convoId.strip():  # Check for non-empty string
                messages = self.convos[convoId]
                messages.append({
                    "role": "user",
                    "content": (usrPrompt + data),
                })
            else:  # Handle one-shot case
                messages = [
                    {
                        "role": "system",
                        "content": sysPrompt,
                    },
                    {
                        "role": "user",
                        "content": (usrPrompt + data),
                    },
                ]
        except KeyError:  # This should only happen if convoId is provided but doesn't exist
            messages = [
                {
                    "role": "system",
                    "content": sysPrompt,
                },
                {
                    "role": "user",
                    "content": (usrPrompt + data),
                },
            ]
        finally:
            logger.debug(f"Messages: {str(messages)}")
            return messages


    def createGooglePayload(self, sysPrompt="", usrPrompt="", convoId="",data=""):
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
        
        
    def validate_response(self, response_str: str, expected_type):
        """
        Parse and validate an LLM response string against an expected Pydantic model.
        Returns a new instance of expected_type with INVALID values if validation fails.
        
        Args:
            response_str: The string response from the LLM
            expected_type: The Pydantic model class to validate against
        Returns:
            The validated Pydantic model instance, or a new instance with INVALID values if validation fails
        """
        # If response is empty or None, return default INVALID instance
        if not response_str:
            logger.warning("Empty response received")
            return expected_type()
            
        try:
            #remove '''json if it exists
            response_str = response_str.replace('```', '')
            response_str = response_str.replace('json', '')

            # Try to parse the response as JSON
            if isinstance(response_str, str):
                try:
                    parsed_response = json.loads(response_str)
                    logger.debug(f"Successfully parsed JSON response: {parsed_response}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e} \n Need to use regex to parse JSON")
                    logger.warning(f"Raw response: {response_str}")
                    # Try to extract JSON from the string if it's embedded in other text
                    import re
                    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                    if json_match:
                        try:
                            parsed_response = json.loads(json_match.group(0))
                            logger.debug(f"Successfully extracted and parsed JSON from text: {parsed_response}")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse extracted JSON")
                            return expected_type()
                    else:
                        logger.warning("No JSON found in response text")
                        return expected_type()
            else:
                parsed_response = response_str
                
            try:
                # Validate against the expected Pydantic model
                validated_response = expected_type.model_validate(parsed_response)
                return validated_response
            except ValidationError as e:
                logger.warning(f"Pydantic validation error: {str(e)}")
                logger.warning(f"Failed validation for data: {parsed_response}")
                return expected_type()
                
        except Exception as e:
            logger.warning(f"Unexpected validation error: {str(e)}")
            return expected_type()