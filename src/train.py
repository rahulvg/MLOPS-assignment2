import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from model import SimpleCNN

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

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1} Loss:{total_loss:.3f} Acc:{acc:.3f}")

torch.save(model.state_dict(), "models/model.pt")
print("Model saved → models/model.pt")
