from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import os
import pytesseract
import pytesseract

# Manually set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"

# Debug: Print the Tesseract path
print("Tesseract path:", pytesseract.pytesseract.tesseract_cmd)
app = Flask(__name__)

# Load Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model
class CNN_CAPTCHA(torch.nn.Module):
    def __init__(self, num_classes=36):
        super(CNN_CAPTCHA, self).__init__()
        self.cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, num_classes)

    def forward(self, x):
        return self.cnn(x)

cnn_model = CNN_CAPTCHA(num_classes=36).to(device)
cnn_model.load_state_dict(torch.load("cnn_captcha.pth", map_location=device))
cnn_model.eval()

# ViT Model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
vit_model.load_state_dict(torch.load("vit_captcha.pth", map_location=device))
vit_model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_text_ocr(image_path):
    """
    Extract text from CAPTCHA using Tesseract OCR.
    """
    image = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(image, config="--psm 6")  # Page Segmentation Mode 6
    return text.strip()

char_map = "abcdefghijklmnopqrstuvwxyz0123456789"

def predict_cnn(image_path):
    """Predict CAPTCHA using the CNN model."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_pred = cnn_model(image)  # Get logits
        cnn_pred = torch.softmax(cnn_pred, dim=1)  # Apply softmax
        cnn_pred_indices = cnn_pred.argmax(dim=1).cpu().numpy()  # Get predicted character indices
        cnn_text = "".join([char_map[i] for i in cnn_pred_indices])  # Map indices to characters

    return cnn_text.strip()

def extract_text_vit(image_path):
    """Use OCR for ViT-based text extraction."""
    return extract_text_ocr(image_path)

def predict_captcha(image_path):
    """Predict CAPTCHA using CNN, ViT, and OCR as a fallback."""
    cnn_text = predict_cnn(image_path)
    vit_text = extract_text_vit(image_path)
    ocr_text = extract_text_ocr(image_path)

    # Return the first available valid result
    if cnn_text==vit_text:
        return cnn_text
    else:
        return vit_text
    if cnn_text=="" and vit_text=="":
        return ocr_text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        image = request.files["captcha"]
        image_path = "uploaded_captcha.png"
        image.save(image_path)
        prediction = predict_captcha(image_path)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

