from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms
from src.model import SimpleCNN

app = FastAPI()

model = SimpleCNN()
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = output.item()

    label = "dog" if prob > 0.5 else "cat"

    return {
        "label": label,
        "probability": float(prob)
    }
