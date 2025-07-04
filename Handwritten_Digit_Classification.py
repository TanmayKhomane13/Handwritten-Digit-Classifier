import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import cv2

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy


# ================ helper functions ===========================
def show_samples(dataset, num_samples = 5):
    plt.figure(num = "Training set", figsize = (10, 2))

    for i in range(num_samples):
        image, label = dataset[i]
        image = image.view(28, 28)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap = 'gray')
        plt.title(f"label: {label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_predictions(model, dataset, num_samples = 5):
    model.eval()
    plt.figure(num = "Testing set", figsize = (10, 2))

    for i in range(num_samples):
        image, label = dataset[i]
        input_img = image.view(-1, 28 * 28)
        output = model(input_img)
        _, predicted = torch.max(output, 1)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.view(28, 28), cmap = 'gray')
        plt.title(f"GT: {label} | Pred: {predicted.item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_custom_image_cv(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    img_resized = cv2.resize(img_thresh, (28, 28))
    img_normalized = img_resized.astype('float32') / 255.0

    img_tensor = torch.tensor(img_normalized).view(1, 1, 28, 28)
    img_tensor = img_tensor.view(-1, 28 * 28)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    plt.figure(num = "Your image")
    plt.imshow(img_resized, cmap = 'gray')
    plt.title(f"Predicted Digit: {predicted.item()}")
    plt.axis('off')
    plt.show()

    return predicted.item()
# ==================================================================

class BetterClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BetterClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# training sets
train_dataset = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor())

show_samples(train_dataset)

# data loaders
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1000)

# initialize model, loss, optimizer
input_dim = 28 * 28
output_dim = 10

model = BetterClassifier(input_dim, 128, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# training loop
n_epochs = 10

for epoch in range(n_epochs):
    for images, labels in train_loader:
        images = images.view(-1, 28 * 28)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch + 1} / {n_epochs}], Loss: {loss.item():.4f}")

show_predictions(model, test_dataset)

predict_custom_image_cv(model, '/Users/tanmaykhomane/Desktop/OpenCV Workspace/OpenCV/Module 3/one.png')

calculate_accuracy(model, test_loader)
    