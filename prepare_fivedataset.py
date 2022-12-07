from torchvision import datasets,transforms
import torch
import os
import numpy as np
from tqdm import tqdm
import pickle

class notMNIST(torch.utils.data.Dataset):
    """The notMNIST dataset is a image recognition dataset of font glypyhs for the letters A through J useful with simple neural networks. It is quite similar to the classic MNIST dataset of handwritten digits 0 through 9.

    Args:
        root (string): Root directory of dataset where directory ``Traffic signs`` exists.
        split (string): One of {'train', 'test'}.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.

    """

    def __init__(self, root, train=True,transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.filename = "notmnist.zip"
        self.url = "https://github.com/nkundiushuti/notmnist_convert/blob/master/notmnist.zip?raw=true"

        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                # self.download()

        training_file = 'notmnist_train.pkl'
        testing_file = 'notmnist_test.pkl'
        if train:
            with open(os.path.join(root,training_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # train  = u.load()
                train = pickle.load(f)
            self.data = train['features'].astype(np.uint8)
            self.labels = train['labels'].astype(np.uint8)
        else:
            with open(os.path.join(root,testing_file),'rb') as f:
                # u = pickle._Unpickler(f)
                # u.encoding = 'latin1'
                # test  = u.load()
                test = pickle.load(f)

            self.data = test['features'].astype(np.uint8)
            self.labels = test['labels'].astype(np.uint8)


    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img[0])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def download(self):
        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


def load_cifar10():
    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]
    train=datasets.CIFAR10('./data/Five_data/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test=datasets.CIFAR10('./data/Five_data/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    return train, test

def load_mnist():
    mean=(0.1,) # Mean and std including the padding
    std=(0.2752,)
    train=datasets.MNIST('./data/Five_data/',train=True,download=True,transform=transforms.Compose([
        transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test=datasets.MNIST('./data/Five_data/',train=False,download=True,transform=transforms.Compose([
        transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
    return train, test

def load_svhn():
    mean=[0.4377,0.4438,0.4728]
    std=[0.198,0.201,0.197]
    train=datasets.SVHN('./data/Five_data/',split='train',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test=datasets.SVHN('./data/Five_data/',split='test',download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    return train, test

def load_fashion_mnist():
    mean=(0.2190,) # Mean and std including the padding
    std=(0.3318,)
    train=datasets.FashionMNIST('./data/Five_data/fashion_mnist',train=True,download=True,transform=transforms.Compose([
        transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test=datasets.FashionMNIST('./data/Five_data/fashion_mnist',train=False,download=True,transform=transforms.Compose([
        transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
    return train, test

def load_notmnist():
    mean=(0.4254,)
    std=(0.4501,)
    train=notMNIST('./data/Five_data/notmnist', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test=notMNIST('./data/Five_data/notmnist', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    return train, test

five_dataset_train_x = []
five_dataset_train_y = []
five_dataset_test_x = []
five_dataset_test_y = []

train, test = load_cifar10()
print('Read cifar10')
for data in tqdm(train):
    image = np.array(data[0])
    image = np.transpose(image, (1, 2, 0))
    five_dataset_train_x.append(image)
    five_dataset_train_y.append(data[1] + 0)
for data in tqdm(test):
    image = np.array(data[0])
    image = np.transpose(image, (1, 2, 0))
    five_dataset_test_x.append(image)
    five_dataset_test_y.append(data[1] + 0)

train, test = load_mnist()
print('Read mnist')
for data in tqdm(train):
    image = np.array(data[0])
    image = image.repeat(3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    five_dataset_train_x.append(image)
    five_dataset_train_y.append(data[1] + 10)
for data in tqdm(test):
    image = np.array(data[0])
    image = image.repeat(3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    five_dataset_test_x.append(image)
    five_dataset_test_y.append(data[1] + 10)

train, test = load_svhn()
print('Read svhn')
for data in tqdm(train):
    image = np.array(data[0])
    image = np.transpose(image, (1, 2, 0))
    five_dataset_train_x.append(image)
    five_dataset_train_y.append(data[1] + 20)
for data in tqdm(test):
    image = np.array(data[0])
    image = np.transpose(image, (1, 2, 0))
    five_dataset_test_x.append(image)
    five_dataset_test_y.append(data[1] + 20)

train, test = load_fashion_mnist()
print('Read fashion_mnist')
for data in tqdm(train):
    image = np.array(data[0])
    image = image.repeat(3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    five_dataset_train_x.append(image)
    five_dataset_train_y.append(data[1] + 30)
for data in tqdm(test):
    image = np.array(data[0])
    image = image.repeat(3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    five_dataset_test_x.append(image)
    five_dataset_test_y.append(data[1] + 30)

train, test = load_notmnist()
print('Read notmnist')
for data in tqdm(train):
    image = np.array(data[0])
    image = image.repeat(3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    five_dataset_train_x.append(image)
    five_dataset_train_y.append(data[1] + 40)
for data in tqdm(test):
    image = np.array(data[0])
    image = image.repeat(3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    five_dataset_test_x.append(image)
    five_dataset_test_y.append(data[1] + 40)

five_dataset_train_x = np.array(five_dataset_train_x)
five_dataset_train_y = np.array(five_dataset_train_y)
five_dataset_train = {'data': five_dataset_train_x, 'fine_labels': five_dataset_train_y}
pickle.dump(five_dataset_train, open('./data/Five_data/five_dataset_train.pkl', 'wb'))

five_dataset_test_x = np.array(five_dataset_test_x)
five_dataset_test_y = np.array(five_dataset_test_y)
five_dataset_test = {'data': five_dataset_test_x, 'fine_labels': five_dataset_test_y}
pickle.dump(five_dataset_test, open('./data/Five_data/five_dataset_test.pkl', 'wb'))

    
