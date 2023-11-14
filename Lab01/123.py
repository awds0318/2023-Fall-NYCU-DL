import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import model

np.random.seed(42)
train_load = np.loadtxt("./data/train.csv", delimiter=",", dtype="int")
test_data = np.loadtxt("./data/test.csv", delimiter=",", dtype="int")

train_data = train_load[:, 1:]
train_label = train_load[:, 0]

print("shape of train_data: {}".format(train_data.shape))
print("shape of train_label: {}".format(train_label.shape))
print("shape of test_data: {}".format(test_data.shape))
train_image_num = train_data.shape[0]
test_image_num = test_data.shape[0]

print("shape of train_data: {}".format(train_data.shape))
print("shape of train_label: {}".format(train_label.shape))
print("shape of test_data: {}".format(test_data.shape))


print("train_image_num  is : {}".format(train_image_num))
print("test_image_num   is : {}".format(test_image_num))
val_image_num = 4800
label_temp = np.zeros((train_image_num, 10), dtype=np.float32)
for i in range(train_image_num):
    label_temp[i][train_label[i]] = 1
train_label_onehot = np.copy(label_temp)
print("One-hot training labels shape:", train_label_onehot.shape)
# EPOCH = 100
# Batch_size = 1000
EPOCH = 5000
Batch_size = 20
lr = 0.001
net = model.Network()

train_batch_num = (train_image_num - val_image_num) // Batch_size
val_batch_num = (val_image_num) // Batch_size
# test_batch_num = test_image_num // Batch_size

for epoch in range(1, EPOCH + 1):
    train_hit = 0
    val_hit = 0
    total_train_loss = 0
    total_val_loss = 0
    for it in range(train_batch_num):
        pred, train_loss = net.forward(train_data[it * Batch_size : (it + 1) * Batch_size], train_label_onehot[it * Batch_size : (it + 1) * Batch_size])
        pred_index = np.argmax(pred, axis=1)
        train_hit += (pred_index == train_label[it * Batch_size : (it + 1) * Batch_size]).sum()
        total_train_loss += train_loss
        # print("pred", pred)
        # print("loss", train_loss)
        net.backward()
        net.update(lr)
    # print(pred)
    # print((total_train_loss))
    for titt in range(val_batch_num):
        tit = train_batch_num + titt
        pred, val_loss = net.forward(train_data[tit * Batch_size : (tit + 1) * Batch_size], train_label_onehot[tit * Batch_size : (tit + 1) * Batch_size])
        pred_index = np.argmax(pred, axis=1)
        val_hit += (pred_index == train_label[tit * Batch_size : (tit + 1) * Batch_size]).sum()
        total_val_loss += val_loss

    print(
        "Epoch:%3d" % epoch,
        "|Train Loss:%8.4f" % (total_train_loss / train_batch_num),
        "|Train Acc:%3.4f" % (train_hit / (train_image_num - val_image_num) * 100.0),
        "|Val Loss:%8.4f" % (total_val_loss / val_batch_num),
        "|Val Acc:%3.4f" % (val_hit / val_image_num * 100.0),
    )
test_pred_list = []

for tit in range(test_image_num // Batch_size):
    pred, test_loss = net.forward(test_data[tit * Batch_size : (tit + 1) * Batch_size], train_label_onehot[tit * Batch_size : (tit + 1) * Batch_size])
    pred_index = np.argmax(pred, axis=1)
    test_pred_list += pred_index.tolist()


print("Dump file...")
df = pd.DataFrame(test_pred_list, columns=["Category"])
df.to_csv(f"DL_ep{EPOCH}_bat{Batch_size}_lr{lr}.csv", index=True, index_label="Id")
