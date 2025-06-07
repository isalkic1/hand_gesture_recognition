import os
import urllib.request

model_url = "https://pjreddie.com/media/files/cross-hands.weights"
model_path = "models/cross-hands.weights"

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists(model_path):
    print("Downloading model weights...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")
else:
    print("Weights already exist.")
