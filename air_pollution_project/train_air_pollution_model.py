# =========================================================
# Air Pollution Image Detection Model (3-Class CNN)
# =========================================================
# ✅ Future-proof: uses new PyTorch weights API (no warnings)
# ✅ Automatic dataset fix + split
# ✅ Training, validation, testing
# and visual outputs (loss, accuracy, confusion, Grad-CAM)

import os
import shutil
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = r"Air Pollution Image Dataset"
OUTPUT_DIR = "outputs"
SPLIT_DIR = "data_split"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

torch.manual_seed(SEED)
random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

# =========================================================
# 0. AUTO-FIX DATASET EXTENSIONS
# =========================================================
def fix_dataset_extensions(dataset_root):
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
    renamed = 0
    for root, _, files in os.walk(dataset_root):
        for f in files:
            old = Path(root) / f
            if old.suffix.lower() not in valid_exts:
                new = old.with_suffix('.jpg')
                try:
                    os.rename(old, new)
                    renamed += 1
                except Exception:
                    pass
    print(f"✅ Checked & fixed {renamed} invalid extensions (if any).")

fix_dataset_extensions(DATA_DIR)

# =========================================================
# 1. AUTO SPLIT DATASET
# =========================================================
def auto_split_dataset(base_dir=SPLIT_DIR):
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    if all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        print("✅ Dataset already split.")
        return train_dir, val_dir, test_dir

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = [c for c in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, c)) and c.lower() != "metadata"]
    print(f"📂 Found classes: {classes}")

    for c in classes:
        src = os.path.join(DATA_DIR, c)
        imgs = [f for f in os.listdir(src)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
        if not imgs:
            print(f"⚠️ No valid images in '{c}', skipping.")
            continue
        random.shuffle(imgs)
        n = len(imgs)
        n_test, n_val = int(TEST_SPLIT*n), int(VAL_SPLIT*n)
        n_train = n - n_test - n_val
        splits = {train_dir: imgs[:n_train],
                  val_dir: imgs[n_train:n_train+n_val],
                  test_dir: imgs[n_train+n_val:]}
        for split_dir, flist in splits.items():
            dst_dir = os.path.join(split_dir, c)
            os.makedirs(dst_dir, exist_ok=True)
            for f in flist:
                shutil.copy2(os.path.join(src, f), os.path.join(dst_dir, f))
    print("✅ Dataset split into train/val/test.")
    return train_dir, val_dir, test_dir

train_dir, val_dir, test_dir = auto_split_dataset()

# =========================================================
# 2. DATALOADERS
# =========================================================
tf_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])
tf_eval = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=tf_train)
val_data   = datasets.ImageFolder(val_dir,   transform=tf_eval)
test_data  = datasets.ImageFolder(test_dir,  transform=tf_eval)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

classes = train_data.classes
print(f"✅ Classes: {classes}")

# =========================================================
# 3. MODEL
# =========================================================
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================================================
# 4. TRAINING
# =========================================================
train_losses, val_losses, val_accs = [], [], []
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    tr_loss = loss_sum/len(train_loader)

    model.eval()
    correct, total, vloss = 0,0,0
    with torch.no_grad():
        for imgs,lbls in val_loader:
            imgs,lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out,lbls)
            vloss += loss.item()
            pred = out.argmax(1)
            correct += (pred==lbls).sum().item()
            total += lbls.size(0)
    acc = correct/total
    val_loss = vloss/len(val_loader)

    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    val_accs.append(acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train:{tr_loss:.4f} | Val:{val_loss:.4f} | Acc:{acc:.4f}")

    if acc>best_acc:
        best_acc=acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"best_model.pth"))
        print("✅ Saved best model.")

# =========================================================
# 5. TRAINING PLOTS
# =========================================================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses,label='Train Loss'); plt.plot(val_losses,label='Val Loss'); plt.legend(); plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(val_accs,label='Val Acc'); plt.legend(); plt.title("Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR,"training_curves.png"))
plt.show()

# =========================================================
# 6. TEST EVALUATION
# =========================================================
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR,"best_model.pth")))
model.eval()
y_true,y_pred=[],[]
with torch.no_grad():
    for imgs,lbls in test_loader:
        imgs,lbls=imgs.to(device),lbls.to(device)
        preds=model(imgs).argmax(1)
        y_true+=lbls.cpu().numpy().tolist()
        y_pred+=preds.cpu().numpy().tolist()

cm=confusion_matrix(y_true,y_pred)
ConfusionMatrixDisplay(cm,display_labels=classes).plot(cmap='Blues')
plt.title("Confusion Matrix"); plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png")); plt.show()

print("\nClassification Report:")
print(classification_report(y_true,y_pred,target_names=classes))

# =========================================================
# 7. CLASS-WISE ACCURACY
# =========================================================
accs=cm.diagonal()/cm.sum(axis=1)
for c,a in zip(classes,accs): print(f"{c}: {a*100:.2f}%")
plt.bar(classes,accs*100); plt.ylim(0,100); plt.title("Class-wise Accuracy (%)")
plt.savefig(os.path.join(OUTPUT_DIR,"classwise_accuracy.png")); plt.show()

# =========================================================
# 8. FIXED GRAD-CAM
# =========================================================
def generate_gradcam(model,img_path,target_layer,class_names):
    tf=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    img=Image.open(img_path).convert('RGB')
    x=tf(img).unsqueeze(0).to(device)
    model.eval()
    acts,grads={},{}

    def fwd(m,i,o): acts['val']=o
    def bwd(m,gi,go): grads['val']=go[0]

    for n,m in model.named_modules():
        if n==target_layer:
            m.register_forward_hook(fwd)
            m.register_backward_hook(bwd)
            break

    out=model(x)
    pred=out.argmax(1).item()
    model.zero_grad()
    out[0,pred].backward()

    g=grads['val']; a=acts['val']
    w=torch.mean(g,dim=(2,3),keepdim=True)
    cam=torch.sum(w*a,dim=1).squeeze().cpu().numpy()
    cam=np.maximum(cam,0); cam/=cam.max()+1e-8

    imgcv=np.array(img)[:,:,::-1]
    hm=cv2.resize(cam,(imgcv.shape[1],imgcv.shape[0]))
    hm=np.uint8(255*hm); hm=cv2.applyColorMap(hm,cv2.COLORMAP_JET)
    overlay=cv2.addWeighted(imgcv,0.6,hm,0.4,0)
    cv2.putText(overlay,f"Pred: {class_names[pred]}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    outp=os.path.join(OUTPUT_DIR,"gradcam_result.jpg")
    cv2.imwrite(outp,overlay)
    print(f"🧠 Grad-CAM saved at {outp}")

sample=None
for r,_,fs in os.walk(test_dir):
    for f in fs:
        if f.lower().endswith(('.jpg','.png','.jpeg')):
            sample=os.path.join(r,f); break
    if sample: break

if sample: generate_gradcam(model,sample,"layer4",classes)
else: print("⚠️ No test image found for Grad-CAM.")
