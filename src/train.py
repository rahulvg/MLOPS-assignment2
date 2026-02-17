import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import SimpleCNN
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

mlflow.set_experiment("cats_dogs_baseline")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

dataset = ImageFolder("data/raw", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = SimpleCNN()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with mlflow.start_run():

    mlflow.log_param("batch_size", 16)
    mlflow.log_param("lr", 0.001)
    mlflow.log_param("epochs", 2)

    all_preds = []
    all_labels = []

    for epoch in range(2):
        total_loss = 0
        correct = 0

        for images, labels in loader:
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.detach().numpy())
            all_labels.extend(labels.detach().numpy())

        acc = correct / len(dataset)
        print(f"Epoch {epoch+1} Loss:{total_loss:.3f} Acc:{acc:.3f}")

        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("loss", total_loss, step=epoch)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    mlflow.log_artifact("models/model.pt")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.savefig("cm.png")
    mlflow.log_artifact("cm.png")

print("Training logged to MLflow")
