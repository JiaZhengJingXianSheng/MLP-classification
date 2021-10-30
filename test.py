import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


def plt_image():
    net1 = torch.load("./MLP1.pth")

    image_dir = 'lufei.png'
    image = Image.open(image_dir)
    transform = transforms.Compose(
        [transforms.Resize([400, 400]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )
    dict = {
        0: "lufei", 1: "luobin", 2: "namei", 3: "qiaoba", 4: "shanzhi", 5: "suolong", 6: "wusopu"
    }

    image1 = transform(image)
    image1 = image1.unsqueeze(0)
    image1 = image1.to("cuda:0")
    label = net1(image1)

    end = label.argmax(dim=1)
    print("predict result is: " + str(dict[end.item()]))

    plt.imshow(image)
    plt.show()


plt_image()
