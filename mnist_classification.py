"""
IIIIIIIIIII
    III
    III
    III
    III
IIIIIIIIII
    III
    III
    III
    III
IIIIIIIIII
    III
    III
    III
    III
"""


"""
Task-0: Download the MNIST dataset using torchvision. Split data into train, test, and validation.
Apply the following augmentations to images: RandomRotation, RandomCrop, ToTensor, and Normalize. [2 pts]
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import argparse
import numpy as np
import tqdm

# Initialize the parser
parser = argparse.ArgumentParser(description="MNIST Classification")

# Adding optional argument
parser.add_argument("--epochs", type=int, default=5, help="number of epochs for train")
parser.add_argument("--batch_size", type=int, default=64, help="size of batch for train, test and validation dataset")
parser.add_argument("--hidden_layers", nargs='+', type=int, default=[256, 512], help="list of hidden layer sizes")
parser.add_argument("--activation", default="relu", choices=["relu", "sigmoid"], help="define activation function")
parser.add_argument("--loss_function", default="cross_entropy", choices=["cross_entropy", "mse"], help="define loss function")
parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"], help="define optimizer")

# Read arguments from the command line
args = parser.parse_args()

# activation function
if args.activation == "relu":
    activation_fn = nn.ReLU()
elif args.activation == "sigmoid":
    activation_fn = nn.Sigmoid()

# loss function
if args.loss_function == "cross_entropy":
    loss_fn = nn.CrossEntropyLoss()
elif args.loss_function == "mse":
    loss_fn = nn.MSELoss()

# optimizer
if args.optimizer == "adam":
    optimizer = optim.Adam
elif args.optimizer == "sgd":
    optimizer = optim.SGD

transformation = transforms.Compose([
    transforms.RandomRotation(11),
    transforms.RandomCrop(28, padding=4),
    transforms.Normalize((0.1307), (0.3081)) #(mean,std) for whole dataset
])

## 0.1 -> download the mnist datasest using torchvision
# train Dataset
train = datasets.MNIST(
    root="\data",
    train=True,
    # transform=transformation,
    transform=transforms.ToTensor(),
    download=True
)
# test dataset
test_dataset = datasets.MNIST(
    root="\data",
    train=False,
    # transform=transformation,
    transform=transforms.ToTensor(),
    download=True
)

## 0.2 -> split data into train, test and validation

train_dataset, val_dataset, test_dataset = random_split(train, [int(0.7 * len(train)),int(0.2 * len(train)),int(0.1 * len(train))])
# train_dataset, val_dataset= random_split(train, [int(0.8 * len(train)),int(0.2 * len(train))])
# train_dataset, val_dataset, test_dataset= random_split(test_dataset, [8500,1000,500])

class Apply_Transformation_To_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

train_dataset = Apply_Transformation_To_Dataset(train_dataset,transform=transformation)
val_dataset = Apply_Transformation_To_Dataset(val_dataset,transform=transformation)
test_dataset = Apply_Transformation_To_Dataset(test_dataset,transform=transformation)

# print("-"*100+"\ntrain dataset\n",train_dataset.dataset)
# print("-"*100+"\nvalidation dataset\n",val_dataset.dataset)
# print(f'{"-"*100}"\ntest dataset\/",{test_dataset}')
# print(f'{"-"*100}"\ntrain data[0] = {train_dataset.dataset[0][0].shape}\n{train_dataset.dataset[0][0]}')
# print("-"*100+"\ntrain data label[1]\n",train_dataset.dataset[0][1])


"""
Task-1: Plot a few images from each class. Create a data loader for the
training dataset as well as the testing dataset. [2 pts]
"""
### 1.1 -> plot few image from class
def plot_images_by_class(dataset, num_images_per_class=9):
    # getting uniques classes of  MNIST dataset
    unique_labels = sorted(set(target for _, target in dataset))
    fig, axes = plt.subplots(nrows=len(unique_labels), ncols=num_images_per_class)
    plt.subplots_adjust(left=0.071,bottom=0.00,right=0.926,top=0.81,wspace=0.02, hspace=0.75)
    fig.suptitle("MNIST Classes")
    for i, label in enumerate(unique_labels):
        indices = [j for j, (_, target) in enumerate(dataset) if target == label][:num_images_per_class]
        for j, ax in enumerate(axes[i]):
            if j < len(indices):
                img, _ = dataset[indices[j]]
                ax.imshow(img.squeeze(), cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')
            if j == len(unique_labels)//2-1:
                ax.text(-15, -6, 'CLASS - ' + str(label),color='blue',fontsize=9)
    plt.show()

plot_images_by_class(test_dataset)
### 1.2 -> create data loader form traing dataset as well as test dataset
batchs = args.batch_size
mnist = {
    'train_dataset' : DataLoader(dataset=train_dataset,batch_size=batchs,shuffle=True),
    'vaildation_dataset' : DataLoader(dataset=val_dataset,batch_size=batchs,shuffle=True),
    'test_dataset': DataLoader(dataset=test_dataset,batch_size=batchs,shuffle=True),
}

"""
Task-2: Write a 3-Layer MLP using PyTorch all using Linear layers. 
Print the number of trainable parameters of the model. [4 pts]
"""

### 2.1 -> 3-Layer MLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"suitable device: {device}")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        # adding hidden layers
        for size in hidden_layers[:-1]:
            # self.layers.append(activation)
            self.layers.append(nn.Linear(size, hidden_layers[1]))        
        # adding output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        # adding activation function to output layer
        self.output_activation = nn.Softmax(dim=1) if output_size > 1 else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_activation(x)

# define input size, output size and hidden layers
input_size = mnist['train_dataset'].dataset[0][0].shape[1:][0]* mnist['train_dataset'].dataset[0][0].shape[1:][1]
output_size = 10
hidden_layers = args.hidden_layers

# MLP model
model = MLP(input_size, hidden_layers, output_size).to(device)


### 2.2 -> Print the number of trainable parameters of the model
trainable_params = sum(p.numel() for p in model.parameters())
print(model,'\n\n','-'*151)
print(f"Trainable Parameters: {trainable_params}")
print('-'*151)

"""
Task-3: Train the model for 5 epochs using Adam as the optimizer and CrossEntropyLoss as the Loss 
Function. Make sure to evaluate the model on the validation set after each epoch and save 
the best model as well as log the accuracy and loss of the model on training and validation data 
at the end of each epoch. [4 pts]
"""
train_losses, val_losses, train_accuracy, val_accuracy = [], [], [], []
num_epochs = 5
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optimizer(model.parameters(), lr=learning_rate)
best_val_loss = float('inf')

for epoch in tqdm.tqdm(range(num_epochs),desc='Epochs : '):
    # train loop
    model.train()
    train_loss_ = 0.0
    for t_image, t_labels in tqdm.tqdm(mnist['train_dataset'], desc='Training : '):
        t_image = t_image.to(device)
        t_labels = t_labels.to(device)
        optimizer.zero_grad()
        t_outputs = model(t_image.view(t_image.size(0), -1))
        t_labels = torch.Tensor(t_labels)
        t_loss = criterion(t_outputs, t_labels)
        t_loss.backward()
        optimizer.step()
        train_loss_ += t_loss.item() * t_image.size(0)
    # average train loss
    avg_train_loss = train_loss_ / len(mnist['train_dataset'])
    train_acc = accuracy_score(t_labels.cpu().numpy(), t_outputs.argmax(dim=1).cpu().numpy())
    train_losses.append(avg_train_loss)
    train_accuracy.append(train_acc)
    
    # Validation loop
    model.eval()
    val_loss_ = 0.0
    with torch.no_grad():
        for v_image, v_labels in tqdm.tqdm(mnist['test_dataset'],desc='Evaluating (on validation dataset) : '):
            v_image = v_image.to(device)
            v_labels = v_labels.to(device)
            v_image = v_image.view(-1, 28*28)
            v_outputs = model(v_image)
            v_labels = torch.Tensor(v_labels)
            v_loss = criterion(v_outputs, v_labels)
            val_loss_ += v_loss.item() * v_image.size(0)
    
    # average validation losses
    avg_val_loss = val_loss_ / len(mnist['vaildation_dataset'])
    val_acc = accuracy_score(v_labels.cpu().numpy(), v_outputs.argmax(dim=1).cpu().numpy())
    val_losses.append(avg_val_loss)
    val_accuracy.append(val_acc)

    # Log accuracy and loss after each epoch
    print(f"\n\nTrain Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}\n\n")
    print('-.-'*51,'\n')
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

"""
Task-4: Visualize correct and Incorrect predictions along with Loss-Epoch
and Accuracy-Epoch graphs for both training and validation. [3 pts]
"""
## 4.1 -> Visualizing Correct and Incorrect Predictions

model.eval()
images, labels = next(iter(mnist['vaildation_dataset']))
images, labels = images.to(device), labels.to(device)
outputs = model(images.view(images.size(0), -1))
_, preds = torch.max(outputs, 1)
wrong_pred_img = []
wrong_pred_label = []
correct_label = []
for idx in range(len(preds)):
  if  preds[idx] != labels[idx]:
      wrong_pred_img.append(np.squeeze(images.cpu().numpy()[idx]))
      wrong_pred_label.append(preds[idx].item())
      correct_label.append(labels[idx].item())

# plotting orignal image and labeling orignal class with wrong class
fig = plt.figure()
fig.suptitle("Wrong Predictions",fontsize=16)
for i in range(len(wrong_pred_label)):
  if i < 10:
    ax = plt.subplot(1, 10, i+1)
    ax.set_title(f"orignal: {correct_label[i]}\nPred: {wrong_pred_label[i]}")
    ax.imshow(wrong_pred_img[i], cmap='gray')
    ax.axis('off')
plt.show()

## 4.2 -> Loss-Epoch and Accuracy-Epoch graphs for both training and validation
# Plotting losses
plt.plot(train_losses, '-o', label='Train loss')
plt.plot(val_losses, '-o', label='Validation loss')
plt.title('Losses : Train vs Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting accuracies
plt.plot(train_accuracy, '-o', label='Train accuracy')
plt.plot(val_accuracy, '-o', label='Validation accuracy')
plt.title("Accuracy : Train vs Validation")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

