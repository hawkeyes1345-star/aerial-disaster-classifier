import os, torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = r"C:\flood_project"
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- data ----
eval_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "Test"), transform=eval_tf)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
class_names = test_ds.classes
print("Classes:", class_names)

# ---- model ----
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_model.pth")))
model = model.to(device).eval()
print("Model loaded.\n")

# ---- evaluate ----
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        out = model(imgs.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
all_preds, all_labels = np.array(all_preds), np.array(all_labels)
test_acc = (all_preds == all_labels).mean()
print(f"TEST ACCURACY: {test_acc:.4f}\n")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

# ---- confusion matrix (saved to file) ----
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(7,6))
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
plt.title(f'Confusion Matrix (Test Acc: {test_acc:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "confusion_matrix.png"), dpi=150)
print("\nSaved confusion_matrix.png")

# ---- Grad-CAM (saved to file) ----
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
def denorm(t):
    img = t.cpu().numpy().transpose(1,2,0) * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)
imgs, labels = next(iter(test_loader)); imgs = imgs.to(device)
fig, axes = plt.subplots(2, 4, figsize=(16,8))
for i in range(4):
    rgb = denorm(imgs[i]); gcam = cam(input_tensor=imgs[i:i+1])[0]
    vis = show_cam_on_image(rgb, gcam, use_rgb=True)
    pred = model(imgs[i:i+1]).argmax(1).item(); true = labels[i].item()
    axes[0,i].imshow(rgb); axes[0,i].set_title(f"True: {class_names[true]}"); axes[0,i].axis('off')
    axes[1,i].imshow(vis); axes[1,i].set_title(f"Pred: {class_names[pred]}",
                    color='green' if pred==true else 'red'); axes[1,i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "gradcam.png"), dpi=150)
print("Saved gradcam.png")
print("\nDONE — check C:\\flood_project for the two PNG files.")