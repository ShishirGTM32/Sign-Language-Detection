import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# Set to help debug CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Dataset class
class FilenameBasedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for img_name in os.listdir(root_dir):
            if img_name.endswith(".jpeg"):
                img_path = os.path.join(root_dir, img_name)
                label_char = img_name[0].lower()
                if '0' <= label_char <= '9':
                    label_index = ord(label_char) - ord('0')
                elif 'a' <= label_char <= 'z':
                    label_index = 10 + ord(label_char) - ord('a')
                else:
                    continue
                self.image_paths.append(img_path)
                self.labels.append(label_index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load datasets
train_dataset = FilenameBasedDataset('/media/decoy/myssd/Sign Language Detection/TrainData', transform=train_transform)
test_dataset = FilenameBasedDataset('/media/decoy/myssd/Sign Language Detection/TestData', transform=test_transform)

# Split datasets
train_ds, val_ds = random_split(train_dataset, [1800, len(train_dataset) - 1800])
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Display one image
for images, labels in train_loader:
    img = images[0].permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    plt.imshow(img)
    plt.title(f"Label: {labels[0].item()}")
    plt.show()
    break

# CNN model
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes, device, image_size=128):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            dummy_output = self._forward_features(dummy_input)
            n_features = dummy_output.shape[1]

        self.fc1 = nn.Linear(n_features, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def _forward_features(self, xb):
        xb = self.pool(F.relu(self.bn1(self.conv1(xb))))
        xb = self.pool(F.relu(self.bn2(self.conv2(xb))))
        xb = self.pool(F.relu(self.bn3(self.conv3(xb))))
        xb = self.pool(F.relu(self.bn4(self.conv4(xb))))
        xb = self.dropout(xb)
        return torch.flatten(xb, 1)

    def forward(self, xb):
        xb = xb.to(self.device)
        xb = self._forward_features(xb)
        xb = self.dropout(F.relu(self.bn_fc1(self.fc1(xb))))
        xb = self.dropout(F.relu(self.bn_fc2(self.fc2(xb))))
        return self.fc3(xb)

    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.accuracy(outputs, labels)
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = self.accuracy(outputs, labels)
        return {'val_loss': loss.detach(), 'val_acc': torch.tensor(acc)}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accs = [x['val_acc'] for x in outputs]
        return {
            'val_loss': torch.stack(batch_losses).mean().item(),
            'val_acc': torch.stack(batch_accs).mean().item()
        }

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch+1}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return (preds == labels).float().mean().item()

# Initialize model with 36 classes
model = SignLanguageCNN(num_classes=36, device=device).to(device)

# Evaluation
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Training
def fit(epochs, lr, model, train_loader, val_loader, opt_func=optim.Adam):
    optimizer = opt_func(model.parameters(), lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    history = []
    for epoch in range(epochs):
        model.train()
        losses, accs = [], []
        for batch in train_loader:
            loss, acc = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            accs.append(acc)
        result = evaluate(model, val_loader)
        result['train_loss'] = sum(losses) / len(losses)
        result['train_acc'] = sum(accs) / len(accs)
        model.epoch_end(epoch, result)
        scheduler.step()
        history.append(result)
    return history

# Train and plot
result0 = evaluate(model, val_loader)
history = [result0]
for _ in range(5):
    history += fit(5, 0.01, model, train_loader, val_loader)

# Plot accuracy
accuracies = [x['val_acc'] for x in history]
plt.plot(accuracies, '-x')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. Epochs')
plt.show()

# Prediction
def predict_image(img, model):
    xb = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(xb)
    _, preds = torch.max(outputs, dim=1)
    return preds[0].item()

# Show one prediction
img, label = test_dataset[45]
img_disp = img.permute(1, 2, 0).numpy()
img_disp = (img_disp * 0.5) + 0.5
plt.imshow(img_disp)
plt.title(f"True: {label}, Predicted: {predict_image(img, model)}")
plt.show()

print(f"True: {label}, Predicted: {predict_image(img, model)}")