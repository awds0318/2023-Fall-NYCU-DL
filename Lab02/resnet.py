import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms

# dataset path
data_path_train = "./data/training"
data_path_test = "./data/testing"

# data transform, you can add different transform methods and resize image to any size
img_size = 224
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)

# build dataset
dataset = datasets.ImageFolder(root=data_path_train, transform=transform)

# spilt your data into train and val
TOTAL_SIZE = len(dataset)
ratio = 0.95
train_len = round(TOTAL_SIZE * ratio)
valid_len = round(TOTAL_SIZE * (1 - ratio))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, valid_len])

# build dataloader
batch_size = 64
train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# check dataset
print(dataset)
print(dataset.class_to_idx)


# train function
def train(model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0

    # Iterate over data
    for inputs, labels in train_data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        total_loss += loss.item()
        total_correct += torch.sum(preds == labels.data)

    avg_loss = total_loss / len(train_data_loader)
    accuracy = total_correct.double() / len(train_dataset) * 100

    print("Training Accuracy: {:.4f}% Training Loss: {:.4f}".format(accuracy, avg_loss))
    return


# validation function
def valid(model, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    # Iterate over data
    for inputs, labels in val_data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        total_loss += loss.item()
        total_correct += torch.sum(preds == labels.data)

    avg_loss = total_loss / len(val_data_loader)
    accuracy = total_correct.double() / len(val_dataset) * 100

    print("Validation Accuracy: {:.4f}% Validation Loss: {:.4f}".format(accuracy, avg_loss))
    return accuracy


# using gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if stride != 1 or ch_out != ch_in:
            self.extra = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride), nn.BatchNorm2d(ch_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv2(out))
        # shortcut
        out = self.extra(x) + out
        out = F.relu(out)
        return out


# build your model here
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0), nn.BatchNorm2d(64))
        self.blk1 = ResBlock(64, 128, stride=2)
        self.blk2 = ResBlock(128, 256, stride=2)
        self.blk3 = ResBlock(256, 512, stride=2)
        self.blk4 = ResBlock(512, 512, stride=2)
        self.outlayer = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


# call model
model = ResNet18()


# -----------------  implement your optimizer -----------------------------------
# you can use any training methods if you want (ex:lr decay, weight decay.....)
learning_rate = 0.001
epochs = 50
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# start training
model.to(device=device)
acc_best = 0.0

print("--------------start training--------------")
for epoch in range(1, epochs + 1):
    print("epoch:", epoch)
    train(model, criterion, optimizer)
    accuracy = valid(model, criterion)

    if accuracy > acc_best:
        acc_best = accuracy
        print("model saved")
        # save the model
        torch.save(model, "model.pth")
