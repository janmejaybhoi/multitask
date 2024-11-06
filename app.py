from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
from torch import nn
from transformers import AutoTokenizer
from typing import List
from transformers import AutoTokenizer, AutoModel

class MultitaskModel(nn.Module):
    def __init__(self, model_name, num_product_labels, num_emotion_labels):
        super(MultitaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        
        self.product_classifier = nn.Linear(hidden_size, num_product_labels)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotion_labels)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,  # labels is a tensor of shape (batch_size, 2)
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        
        product_logits = self.product_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)
        
        loss = None
        if labels is not None:
            product_labels = labels[:, 0]
            emotion_labels = labels[:, 1]
            loss_fct = nn.CrossEntropyLoss()
            product_loss = loss_fct(product_logits, product_labels)
            emotion_loss = loss_fct(emotion_logits, emotion_labels)
            loss = product_loss + emotion_loss  # Combine losses
        
        return {
            'loss': loss,
            'logits': (product_logits, emotion_logits)
        }


# Load model, tokenizer, and encoders
product_encoder = joblib.load('MODEL_04/product_encoder.joblib')
emotion_encoder = joblib.load('MODEL_04/emotion_encoder.joblib')
num_product_labels, num_emotion_labels = len(product_encoder.classes_), len(emotion_encoder.classes_)

model = MultitaskModel(
    model_name='bert-base-uncased', 
    num_product_labels=num_product_labels,
    num_emotion_labels=num_emotion_labels,
)

tokenizer = AutoTokenizer.from_pretrained('MODEL_04/')
state_dict = torch.load('MODEL_04/pytorch_model.bin', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

app = FastAPI()

# Define request model
class TextInput(BaseModel):
    texts: List[str]

# Prediction function
def predict_text(texts):
    predictions = []
    emotion_predictions = []
    product_predictions = []

    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        if 'token_type_ids' in inputs and 'token_type_ids' not in model.forward.__code__.co_varnames:
            del inputs['token_type_ids']

        with torch.no_grad():
            outputs = model(**inputs)

            if isinstance(outputs['logits'], tuple):
                product_logits = outputs['logits'][0]
                emotion_logits = outputs['logits'][1]
            else:
                product_logits = outputs['logits'][:, len(product_encoder.classes_):]
                emotion_logits = outputs['logits'][:, :len(emotion_encoder.classes_)]

            emotion_pred_index = torch.argmax(emotion_logits, dim=1).item()
            product_pred_index = torch.argmax(product_logits, dim=1).item()

            emotion_pred_class = emotion_encoder.inverse_transform([emotion_pred_index])[0]
            product_pred_class = product_encoder.inverse_transform([product_pred_index])[0]

        predictions.append({
            "text": text,
            "emotion_prediction": emotion_pred_class,
            "product_prediction": product_pred_class
        })

    return predictions

# API endpoint for predictions
@app.post("/predict")
async def get_predictions(input_data: TextInput):
    try:
        predictions = predict_text(input_data.texts)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
