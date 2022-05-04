import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vgg import VGG

IMG_PATH = 'data/my_test_imgs'
#IMG_PATH = 'data/val/roses'
JSON_PATH = 'class_idx.json'
WEIGHT_PATH = 'vgg19.pth'

#python 的标准库手册推荐在任何情况下尽量使用time.clock().
#只计算了程序运行CPU的时间，返回值是浮点数
import time

def predict(net, img, json_label):
   
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    original_img=img
    img = data_transform(img)  # 3,224,224
    img = torch.unsqueeze(img, dim=0)  # 1,3,224,224
    assert os.path.exists(WEIGHT_PATH), f'file {WEIGHT_PATH} does not exist.'
    net.load_state_dict(torch.load(WEIGHT_PATH))
   
    start =time.process_time()
    net.eval()
  #  end = time.process_time()
    img_temp=net(img)
    end = time.process_time()

    with torch.no_grad():
        output = torch.squeeze(img_temp)  # net(img)的size为1,5，经过squeeze后变为5
        predict = torch.softmax(output, dim=0)
        predict_label_idx=int(torch.argmax(predict))
        predict_label=json_label[str(predict_label_idx)]
        predict_probability=predict[predict_label_idx]
  #  end = time.process_time()

    #print('Running time: %s Seconds'%(end-start))

    predict_result=f'class:{predict_label}, probability:{predict_probability:.3f}'
    plt.imshow(original_img)
    plt.title(predict_result)
    print(predict_result)
    plt.show()
    return (end-start)


def read_json(json_path):
    assert os.path.exists(json_path), f'{json_path} does not exist.'
    with open(json_path, 'r') as json_file:
        idx2class = json.load(json_file)
        return idx2class


if __name__ == '__main__':
    net = VGG(num_labels=5)
    images=os.listdir(IMG_PATH)
    inference_time=[]
    #print(images)
    for index,image in enumerate(images):
        image_path = os.path.join(IMG_PATH, image)
        print(index,image_path)
        img = Image.open(image_path)
        idx2class = read_json(JSON_PATH)
        infer_time=predict(net, img, idx2class)
        inference_time.append(infer_time)
    print("Sum time:", sum(inference_time))
    print("average time:",sum(inference_time)/len(inference_time))
