import torch
from src.preprocess import load_image

def test_image_shape():
    tensor = load_image("50.jpg")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3,224,224)
