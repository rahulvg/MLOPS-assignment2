import requests

test_samples = [
    ("50.jpg", 0),
    ("dog.png", 1),
]

correct = 0

for img_path, true_label in test_samples:
    with open(img_path, "rb") as f:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": f}
        )
        prediction = response.json()["label"]
        pred_label = 1 if prediction == "dog" else 0

        if pred_label == true_label:
            correct += 1

accuracy = correct / len(test_samples)
print("Post-deployment accuracy:", accuracy)