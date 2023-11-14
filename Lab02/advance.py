import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# dataset path
data_path_train = "data/training"
data_path_test = "data/testing"


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

# build your model here
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 4)

# ------------------ implement your optimizer -----------------------------------
# you can use any training methods if you want (ex:lr decay, weight decay.....)
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
epochs = 10

# start training
model.to(device=device)
acc_best = 0.0

print("--------------start training--------------")
for epoch in range(1, epochs + 1):
    print("epoch:", epoch)
    train(model, criterion, optimizer)
    accuracy = valid(model, criterion)
    writer.add_scalar("Learning_rate", learning_rate, epoch)
    writer.add_histogram("Weights", model.conv1.weight, epoch)
    if accuracy > acc_best:
        acc_best = accuracy
        print("model saved")
        torch.save(model, "model.pth")


transform_test = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

dataset_test = datasets.ImageFolder(root=data_path_test, transform=transform_test)
dataloader_test = data.DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4)


# load the model so that you don't need to train the model again
test_model = torch.load("model.pth").to(device)


def test(model):
    with torch.no_grad():
        model.eval()
        bs = dataloader_test.batch_size
        result = []
        for i, (data_test, target) in enumerate(dataloader_test):
            data, target = data_test.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1, keepdim=True)

            preds.data.cpu().numpy()
            for j in range(preds.size()[0]):
                file_name = dataset_test.samples[i * bs + j][0].split("/")[-1]
                result.append((file_name, preds[j].cpu().numpy()[0]))
    return result


result = test(test_model)

with open(f"{epoch}_{learning_rate}_{batch_size}_ID_result.csv", "w") as f:
    f.write("ID,label\n")
    for r in result:
        f.write(r[0] + "," + str(r[1]) + "\n")
