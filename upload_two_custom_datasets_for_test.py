# Upload custom data
# Loading data from google drive
test_data = np.load("/content/drive/My Drive/Yura/NewDataSet_test_data.npy", allow_pickle=True)
new_test_data = np.load("/content/drive/My Drive/Yura/NewDataSet_test_data_new_cancer.npy", allow_pickle=True)
print("Data prepared")

X_test = torch.Tensor([i[0] for i in test_data]).view(-1, 1, IMG_SIZE, IMG_SIZE)
X_test = X_test / 255.0
y_test = torch.Tensor([i[1] for i in test_data])

new_X_test = torch.Tensor([i[0] for i in new_test_data]).view(-1, 1, IMG_SIZE, IMG_SIZE)
new_X_test = new_X_test / 255.0
new_y_test = torch.Tensor([i[1] for i in new_test_data])


# Accuracy test
def test_custom(net):
    global IMG_SIZE
    print("begin the test")
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            real_class = torch.argmax(y_test[i]).to(device)
            net_out = net(X_test[i].view(-1, 1, IMG_SIZE, IMG_SIZE).to(device))[0]

            predicted_class = torch.argmax(net_out)

            # print("Predicted class: ", predicted_class, "Real class: ", real_class)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy: ", round(correct / total, 3))


def new_test_custom(net):
    global IMG_SIZE
    print("begin the test")
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(new_X_test))):
            real_class = torch.argmax(new_y_test[i]).to(device)
            net_out = net(new_X_test[i].view(-1, 1, IMG_SIZE, IMG_SIZE).to(device))[0]

            predicted_class = torch.argmax(net_out)

            # print("Predicted class: ", predicted_class, "Real class: ", real_class)

            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy on NewDataSet_test_data_new_cancer : ", round(correct / total, 3))


test_custom(net)
new_test_custom(net)

print("DONE")