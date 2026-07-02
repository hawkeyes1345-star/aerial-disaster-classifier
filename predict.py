import sys, os, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DATA_DIR = r"C:\flood_project"
CLASS_NAMES = ['Earthquake', 'Fire', 'Flood', 'Normal']
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load trained model ---
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_model.pth")))
model = model.to(device).eval()

# --- preprocessing (same as eval) ---
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                         transforms.Normalize(MEAN, STD)])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"PREDICTION: {CLASS_NAMES[pred_idx]}  ({probs[pred_idx]*100:.1f}% confident)\n")
    print("All class probabilities:")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"  {name:12s} {p*100:5.1f}%  {bar}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1])