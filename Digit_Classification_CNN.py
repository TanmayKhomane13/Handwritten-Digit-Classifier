import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
import cv2

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy
# ================== helper functions =======================
def show_predictions(model, dataset, num_samples = 5):
    model.eval()
    plt.figure(num = "Testing set", figsize = (10, 2))

    for i in range(num_samples):
        image, label = dataset[i]
        input_img = image.unsqueeze(0)
        output = model(input_img)
        _, predicted = torch.max(output, 1)

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.view(16, 16), cmap = 'gray')
        plt.title(f"GT: {label} | Pred: {predicted.item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_custom_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    img_resized = cv2.resize(img_thresh, (16, 16))
    img_normalized = img_resized.astype('float32') / 255.0

    img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0)

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
# ========================================================================

IMAGE_SIZE = 16
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

# for plotting data samples as images
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap = 'gray')
    plt.title('y = ' + str(data_sample[1]))
    plt.show()

# ====================== CNN ===========================
class CNN(nn.Module):
    def __init__(self, out_1 = 16, out_2 = 32):
        super(CNN, self).__init__()
        # The reason we start with 1 channel is because we have a single black and white image
        # Channel Width after this layer is 16
        self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = out_1, kernel_size = 5, padding = 2)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        # Channel Width after this layer is 8
        self.cnn2 = nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 5, stride = 1, padding = 2)
        # Channel Width after this layer is 4
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        # In total we have out_2 (32) channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        self.fc1 = nn.Linear(out_2 * 4 * 4, 10)
    
    # prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        return x
# =========================================================================
def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.data
        
        cost_list.append(COST)
        
        correct = 0

        for x, y in test_loader:
            z = model(x)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y).sum().item()
        
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        print(f"Epoch [{epoch + 1} / {n_epochs}], Loss: {loss.item()}")
# ========================================================================

train_dataset = dsets.MNIST(root = './data', train = True, download = True, transform = composed)
test_dataset = dsets.MNIST(root = './data', train = False, download = True, transform = composed)

model = CNN(out_1 = 16, out_2 = 32)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 5000)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

n_epochs = 5

cost_list = []
accuracy_list = []
N_test = len(test_dataset)

train_model(n_epochs)

# show_predictions(model, test_dataset)

predict_custom_image(model, '/Users/tanmaykhomane/Desktop/OpenCV Workspace/OpenCV/Module 4/nine.png')

calculate_accuracy(model, test_loader)
