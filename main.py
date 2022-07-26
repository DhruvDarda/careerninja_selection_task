from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained(
    'bert-base-uncased', output_hidden_states=True)


def get_embeddings(text, token_length):
    tokens = tokenizer(text, max_length=token_length,
                       padding='max_length', truncation=True)
    output = model(torch.tensor(tokens.input_ids).unsqueeze(0),
                   attention_mask=torch.tensor(tokens.attention_mask).unsqueeze(0)).hidden_states[-1]
    return torch.mean(output, axis=1).detach().numpy()


def calculate_similarity(text1, text2, token_length=20):
    out1 = get_embeddings(text1, token_length=token_length)
    out2 = get_embeddings(text2, token_length=token_length)
    sim = cosine_similarity(out1, out2)[0][0]
    print(sim)


app = FastAPI(title="Similarity Score")


class UserInput(BaseModel):
    user_input: float


@app.get('/calculate/')
async def sentences(sen1: str, sen2: str):

    similarity = calculate_similarity(sen1, sen2)

    return {"calculate": float(similarity)}

uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
