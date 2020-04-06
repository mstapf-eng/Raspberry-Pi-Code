import torch
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.optim import Adam
from flowernet import FlowerClassifierCNNModel
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from util import show_transformed_image


class FlowerModel:
    def __init__(self, data_folder="./data/flowers", cudaready = False):
        self.cuda = cudaready
        self.cudaready = False
        self.cnn_model = FlowerClassifierCNNModel()
        self.optimizer = Adam(self.cnn_model.parameters())
        #self.loss_fn = nn.CrossEntropyLoss()

        if (self.cuda):
            self.cnn_model.cuda()
        if (self.cuda):
            self.loss_fn = nn.CrossEntropyLoss().cuda()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.transformations = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        total_dataset = datasets.ImageFolder(data_folder, transform=self.transformations)
        dataset_loader = DataLoader(dataset=total_dataset, batch_size=100)
        items = iter(dataset_loader)
        self.image, self.label = items.next()

        train_size = int(0.8 * len(total_dataset))
        test_size = len(total_dataset) - train_size
        train_dataset, self.test_dataset = random_split(total_dataset, [train_size, test_size])

        self.train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=100)
        self.test_dataset_loader = DataLoader(dataset=self.test_dataset, batch_size=100)

    def train(self, epoches=10):
        for epoch in range(epoches):
            self.cnn_model.train()
            running_loss = 0
            for i, (self.images, self.labels) in enumerate(self.train_dataset_loader):
                self.optimizer.zero_grad()
                outputs = self.cnn_model(self.images)
                #loss = self.loss_fn(outputs, labels)
                #print("iteration" + str(i) + ":" + str(loss))
                if i % 5 == 4:  # print every 5 mini-batches
                    print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, loss / (i + 1)))
                if (self.cuda):
                    self.images, self.labels = Variable(self.images.cuda(non_blocking=True)), Variable(self.labels.cuda(non_blocking=True))
                else:
                    self.images, self.labels = Variable(self.images), Variable(self.labels)
                self.optimizer.zero_grad()
                if self.cuda:
                    outputs = self.cnn_model(self.images).cuda(non_blocking=True)
                else:
                    outputs = self.cnn_model(self.images)
                outputs = outputs.squeeze()
                loss = self.loss_fn(outputs, self.labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

    ### save model
    def saveModel(self, mfile='model.pth'):
        print('Saving Model ')
        torch.save(self.cnn_model.state_dict(), mfile)

    #### Load model from file system
    def loadModel(self, mfile='model.pth'):
        self.cnn_model.load_state_dict(torch.load(mfile, map_location=lambda storage, loc: storage))

    # My addition - Unsure if this is the correct placement for the method
    def TestAccuracy(self):
        self.cnn_model.eval()
        test_acc_count = 0
        for k, (test_images, test_labels) in enumerate(self.test_dataset_loader):
            test_outputs = self.cnn_model(test_images)
            _, prediction = torch.max(test_outputs.data, 1)
            test_acc_count += torch.sum(prediction == test_labels.data).item()

        test_accuracy = test_acc_count / len(self.test_dataset)
        print("Accuracy percentage: " + str(test_accuracy))

    def predict(self, filename):
        test_image = Image.open(filename)
        test_image_tensor = self.transformations(test_image).float()
        #test_image_tensor = test_image_tensor.unsqueeze_(0)
        #output = self.cnn_model(test_image_tensor)
        if (self.cuda):
            test_image_tensor = Variable(test_image_tensor.cuda(non_blocking=True))
            output = self.cnn_model(test_image_tensor).cuda(non_blocking=True)
        else:
            test_image_tensor = test_image_tensor.unsqueeze_(0)
            output = self.cnn_model(test_image_tensor)
        class_index = output.data.numpy().argmax()
        return class_index
