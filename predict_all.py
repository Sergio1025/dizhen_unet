import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms

class predict:
    def __init__(self) -> None:
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #载入模型
        self.model = smp.Unet(encoder_name="resnet50",encoder_weights="imagenet", in_channels=3, classes=7)
        path = './model.pth'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        # 数据预处理
        self.transform_img = transforms.Compose(
            [
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ]
        )
        self.model.eval()

    def predict_image(self, img_path):
        
        image = Image.open(img_path).convert('RGB') 
        image = self.transform_img(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(image)
            prediction = output.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint32)
            mask = Image.fromarray(prediction)
        return mask


need_rgb = np.array([[34, 167, 132],[64,  67, 135],[ 42, 119, 142],[253, 231,  36],[121, 209,  81],[68,  1, 84]])

def generate_rgb(gray_array):
    # print('gray_array.shape', gray_array.shape) #打印出灰度图的shape
    x, y = gray_array.shape 
    rgb = np.zeros([x, y, 3], int) #扩充为rgb通道
    # gray_array = np.array(gray_array) #化为矩阵形式
    for i in range(x):
        for j in range(y):
            if gray_array[i, j] == 0:  #目标体（原来是白色）
                rgb[i, j, 0] = 34
                rgb[i, j, 1] = 167 
                rgb[i, j, 2] = 132
            else: 
                rgb[i, j, 0] = need_rgb[gray_array[i, j]-1][0]
                rgb[i, j, 1] = need_rgb[gray_array[i, j]-1][1]
                rgb[i, j, 2] = need_rgb[gray_array[i, j]-1][2]
    return rgb

# 需要预测的图片路径
root_image = r'in_image'
# 保存结果的路径
save_mask_root = r'result'

# 定义检测代码
model_predict = predict()

if not os.path.exists(save_mask_root):
    os.makedirs(save_mask_root)

for root, dir, files in os.walk(os.path.join(root_image)):
    for file in tqdm(files):
        
        ori_image_path = os.path.join(root_image, file)

        # 获取预测结果
        predict_mask = model_predict.predict_image(ori_image_path)

        #调用generate_rgb函数上色
        predict_mask = np.array(predict_mask)    #化为矩阵
        predict_mask = generate_rgb(predict_mask)   
        predict_mask = Image.fromarray(np.uint8(predict_mask))

        # 调整为预测尺寸为图片真实尺寸
        image = Image.open(ori_image_path)
        width, height = image.size
        predict_mask = predict_mask.resize((width,height)) 

        # 保存预测结果
        save_mask_path = os.path.join(save_mask_root, file)
        predict_mask.save(save_mask_path)

