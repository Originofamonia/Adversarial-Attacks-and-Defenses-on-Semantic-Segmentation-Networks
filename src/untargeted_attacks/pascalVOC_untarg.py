import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision import datasets, transforms as T
from utils import decode_segmap, IoUAcc, pgd, pgd_steep

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.transforms.functional import normalize
from sklearn.metrics import accuracy_score
import numpy as np

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()

    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    batch_size = 4

    img_size = (520,520) # original input size to the model is (520,520) but all images in dataset are of different sizes in PascalVOC

    trans = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = datasets.VOCSegmentation(r'datasets/PascalVOC', year='2012', image_set='val', download=False, transform=trans,
                                    target_transform=T.Resize(img_size), transforms=None) # Path to be updated for local use.

    X, y, yrep = [], [], []
    for i in range(batch_size):
        num = torch.randint(0,1449,(1,1)).item()
        X.append(dataset[num][0])
        y.append(np.asarray(dataset[num][1]))
        yrep.append(dataset[num][1])
    X, y = torch.stack(X), torch.tensor(y).unsqueeze(1)

    #print(X.size(), y.size())

    net = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None).eval() #Any pre-trained model from pytorch can be made used of.

    class_names = {1:'background', 2:'aeroplane', 3:'bicycle', 4:'bird', 5:'boat', 6:'bottle', 7:'bus', 8:'car', 9:'cat',
                10:'chair', 11:'cow', 12:'diningtable', 13:'dog', 14:'horse', 15:'motorbike', 16:'person', 17:'pottedplant',
                18:'sheep', 19:'sofa', 20:'train', 21:'tvmonitor'}

    yp = net(X)['out']
    m = torch.softmax(yp,1)
    pred = torch.argmax(m,1)
    clean_iou, clean_acc = IoUAcc(y, pred, class_names)
    print(f"clean_iou, clean_acc: {clean_iou, clean_acc}")

    delta1 = pgd(net, X, y, epsilon=0.10, alpha=1e2, num_iter=10) # Various values of epsilon, alpha can be used to play with.
    adv_images = X.float()+ delta1.float()
    ypa1 = net(adv_images)['out']
    n = torch.softmax(ypa1,1) 
    preda1 = torch.argmax(n,1)
    IoUa1, Acca1 = IoUAcc(y, preda1, class_names)
    print(f'IoUa1, Acca1: {IoUa1, Acca1}')
    denormed_x = denorm(X)  # returns [0,1]
    denormed_x = denormed_x.permute(0, 2, 3, 1).detach().cpu().numpy()
    denormed_adv_x = denorm(adv_images)
    denormed_adv_x = denormed_adv_x.permute(0, 2, 3, 1).detach().cpu().numpy()
    y = y.permute(0, 2, 3, 1).detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    preda1 = preda1.detach().cpu().numpy()

    for i, x_clean in enumerate(denormed_x):
        print(x_clean)
        x_adv = denormed_adv_x[i]
        y_true = y[i]
        y_pred_clean = pred[i]
        y_pred_adv = preda1[i]
        fig, axs = plt.subplots(2, 3, figsize=(14, 6))
        axs[0,0].imshow(x_clean)
        axs[0,0].set_title(f'Clean image')
        axs[0,0].axis('off')

        axs[0,1].imshow(x_clean)
        axs[0,1].imshow(y_true, cmap='viridis', alpha=0.5)
        axs[0,1].set_title(f'y_true')
        axs[0,1].axis('off')

        axs[0,2].imshow(x_clean)
        axs[0,2].imshow(y_pred_clean, cmap='viridis', alpha=0.5)
        axs[0,2].set_title(f'Clean y_pred')
        axs[0,2].axis('off')

        axs[1,0].imshow(x_adv)
        axs[1,0].set_title(f'Adv image')
        axs[1,0].axis('off')

        axs[1,1].imshow(x_adv)
        axs[1,1].imshow(y_pred_adv, cmap='viridis', alpha=0.5)
        axs[1,1].set_title(f'Adv y_pred')
        axs[1,1].axis('off')

        img_filename = f'output/{i}_overlay.png'
        fig.savefig(img_filename, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    main()
