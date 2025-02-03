from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from models import CBOW, CBOWTrainer
from Tokenizer import Tokenizer
from Stemmer import Stemmer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tokenizer
tokenizer = Tokenizer()
vocab_file_path = 'data_outputs/vocab.json'
# Load vocabulary
# with open('./data_outputs/vocab.json', 'r') as f:
    # vocab_data = json.load(f)
    # word_to_id = vocab_data.get('word_to_id', {})
    # id_to_word = {int(k): v for k, v in vocab_data.get('id_to_word', {}).items()}
word_to_id, id_to_word = tokenizer.get_lookup_table(vocab_file_path)
# Initialize model and trainer
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_size = 200  # Make sure this matches your trained model
vocab_size = len(word_to_id)
model = CBOW(vocab_size=vocab_size, embedding_dim=embedding_size).to(device)

# Load model weights
model.load_state_dict(torch.load('models/weights.pt', map_location=device))
model.eval()

# Initialize trainer for using its methods
trainer = CBOWTrainer(
    embedding_size=embedding_size,
    device=device,
    word_to_id=word_to_id,
    id_to_word=id_to_word
)

class TextInput(BaseModel):
    text: str

@app.post("/submit")
async def submit_text(input_data: TextInput):
    try:
        # Prepare the input text
        tokens = trainer.prepare_data_minimal(input_data.text, word_to_id, tokenizer)
        if not tokens:
            return {"error": "No valid tokens found in input"}
            
        # Find similar words for each token
        similar_words = []
        for token_id in tokens:
            if token_id in id_to_word:
                word = id_to_word[token_id]
                similar = trainer.find_similar_words(model, word, n=5)
                similar_words.append({
                    "word": word,
                    "similar": similar
                })
        
        return {"results": similar_words}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "API is running"} 