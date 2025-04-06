from torchvision import transforms
from PIL import Image
import numpy as np 
import torch

def Convert2Tensor(x: torch.tensor) -> torch.tensor:
    x_out = x.permute(2, 0, 1)
    x_out = x_out.float()
    x_out /= 255.0
    return x_out

def Normalize(x: torch.tensor) -> torch.tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor(  [0.229, 0.224, 0.225])

    x_out = (x - mean[:, None, None]) / std[:, None, None]

    return x_out
    

# 定义标准化操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 假设你有一个 PIL 图像 img
img = Image.open('000252.jpg')
img_array = np.array(img)
print(img)
print(img_array.shape)

test = Convert2Tensor(torch.tensor(img_array))
print(test.shape)
normalize_out = Normalize(test)
print(normalize_out)


# 应用标准化
img_tensor = transform(img)
print("==============")
print(img_tensor.size())
print(img_tensor)


x = torch.tensor([[[1, 2],
                   [3, 4]
                  ],
                   [[5, 6],
                    [7, 8]
                  ],
                  [[9, 10],
                   [11, 12]
                  ]])

x_out = Normalize(x)

print(x.shape)
print(x_out)