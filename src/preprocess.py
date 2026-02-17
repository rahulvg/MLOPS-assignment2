from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

if __name__ == "__main__":
    tensor = load_image("data/raw/cat/0.jpg")
    print(tensor.shape)
