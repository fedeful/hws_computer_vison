import os
import cv2
import torch
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model



def features():
    new_model = alexnet(pretrained=True)
    new_classifier = nn.Sequential(*list(new_model.classifier.children())[:-1])
    new_model.classifier = new_classifier

    return new_model


def match(name_image):
    basepath = '/home/federico/PycharmProjects/hws_computer_vison/lab08/iamges'
    MAX_SCORE = 10000000
    NAME = ''


    cl = features()
    cl.eval()

    mimage = cv2.imread(name_image, cv2.IMREAD_COLOR)
    mimage = cv2.cvtColor(mimage, cv2.COLOR_BGR2RGB)
    mimage = cv2.resize(mimage, (224, 224))

    mimage = torch.from_numpy(mimage.transpose(2, 0, 1)).float()
    mimage /= 255
    mimage = torch.unsqueeze(mimage,0)

    with torch.no_grad():
        MIMVAL = cl(mimage)
        MIMVAL = MIMVAL.data.cpu().numpy()

    for name in os.listdir(basepath):

        image = cv2.imread(os.path.join(basepath, name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image /= 255
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            IMVAL = cl(image)
            IMVAL = IMVAL.data.cpu().numpy()

            val = np.abs((MIMVAL.astype(np.float128)**2 - IMVAL.astype(np.float128)**2).squeeze())
            val = np.sum(val, 0)
            print(name, val)

            if val < MAX_SCORE:
                NAME = name
                MAX_SCORE = val

    return NAME


if __name__ == '__main__':
    name = match('/home/federico/PycharmProjects/hws_computer_vison/lab08/car2.png')
    print(name)
