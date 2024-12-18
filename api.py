from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import os
import uvicorn
from neuronpedia.np_vector import NPVector
from neuronpedia.requests.base_request import NPRequest
from neuronpedia.requests.steer_request import SteerChatRequest
from fastapi.middleware.cors import CORSMiddleware
import asyncio
# Set API key

from dotenv import load_dotenv

# Load NEURONPEDIA_API_KEY from .env file
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Vector Steering API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only - in production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define enums for validation
class TaskEnum(str, Enum):
    anger = "anger"
    christian_evangelist = "christian_evangelist"
    conspiracy = "conspiracy"
    french = "french"
    london = "london"
    love = "love"
    praise = "praise"
    want_to_die = "want_to_die"
    wedding = "wedding"

class MethodEnum(str, Enum):
    caao = "caao"
    caa = "caa"
    saets = "saets"
    sae = "sae"

# Request model
class SteerInput(BaseModel):
    task: TaskEnum
    method: MethodEnum
    prompt: str = "I think"  # Default prompt
    temperature: float = 1.0
    strength_multiplier: float = 4.0

class SteerChatInput(BaseModel):
    task: TaskEnum
    method: MethodEnum
    message: str
    temperature: float = 1.0
    strength_multiplier: float = 4.0

# Steering request class
class SteerRequest(NPRequest):
    def __init__(self):
        super().__init__("steer")
    
    def steer(
        self,
        model_id: str,
        vectors: list[NPVector],
        prompt: str,
        temperature: float = 0.5,
        n_tokens: int = 80,
        freq_penalty: float = 2,
        seed: int = 16,
        strength_multiplier: float = 4,
        steer_special_tokens: bool = True,
    ):
        features = [
            {
                "modelId": vector.model_id,
                "layer": vector.source,
                "index": vector.index,
                "strength": vector.default_steer_strength,
            }
            for vector in vectors
        ]
        
        payload = {
            "modelId": model_id,
            "features": features,
            "prompt": prompt,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "seed": seed,
            "strength_multiplier": strength_multiplier,
            "steer_special_tokens": steer_special_tokens,
        }
        return self.send_request(method="POST", json=payload)

# Vector fetching functions
all_vectors = NPVector.get_owned()

def fetch_by_label(label: str) -> Optional[NPVector]:
    for vector in all_vectors:
        if vector.label == label:
            return vector
    return None


@app.post("/steer_chat")
async def steer_chat_endpoint(input_data: SteerChatInput):
    # Construct vector label using gemma2bit prefix
    vector_label = f"gemma2bit_{input_data.task.value}_{input_data.method.value}"
    
    # Fetch vector
    vector = fetch_by_label(vector_label)
    if not vector:
        raise HTTPException(
            status_code=404,
            detail=f"Vector not found for label: {vector_label}"
        )
    
    max_attempts = 3
    delay_seconds = 5
    
    for attempt in range(max_attempts):
        try:
            # Process chat steering request with single user message
            response = SteerChatRequest().steer(
                model_id=vector.model_id,
                vectors=[vector],
                default_chat_messages=[{"role": "user", "content": input_data.message}],
                steered_chat_messages=[{"role": "user", "content": input_data.message}],
                temperature=input_data.temperature,
                strength_multiplier=input_data.strength_multiplier,
                n_tokens=80
            )
            print(input_data.message)
            print(response)
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            # If this was the last attempt, raise the HTTP exception
            if attempt == max_attempts - 1:
                print(f"All {max_attempts} attempts failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing chat steering request after {max_attempts} attempts: {str(e)}"
                )
            # Wait before the next attempt
            await asyncio.sleep(delay_seconds)
            print(f"Retrying... (Attempt {attempt + 2} of {max_attempts})")

@app.post("/steer")
async def steer_endpoint(input_data: SteerInput):
    # Construct vector label using the value of the enum instead of the enum itself
    vector_label = f"gemma2b_{input_data.task.value}_{input_data.method.value}"
    
    # Fetch vector
    vector = fetch_by_label(vector_label)
    if not vector:
        raise HTTPException(
            status_code=404,
            detail=f"Vector not found for label: {vector_label}"
        )
    
    max_attempts = 3
    delay_seconds = 5
    
    for attempt in range(max_attempts):
        try:
            # Process steering request
            response = SteerRequest().steer(
                model_id=vector.model_id,
                vectors=[vector],
                prompt=input_data.prompt,
                temperature=input_data.temperature,
                strength_multiplier=input_data.strength_multiplier
            )
            print(response)
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
            # If this was the last attempt, raise the HTTP exception
            if attempt == max_attempts - 1:
                print(f"All {max_attempts} attempts failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing steering request after {max_attempts} attempts: {str(e)}"
                )
            # Wait before the next attempt
            await asyncio.sleep(delay_seconds)
            print(f"Retrying... (Attempt {attempt + 2} of {max_attempts})")


# Run the server when the script is executed directly
if __name__ == "__main__":
    print("Starting Vector Steering API server...")
    print("Server running at http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)