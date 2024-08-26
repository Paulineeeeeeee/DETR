import torch
import json

import os
from PIL import Image

import torch
import torchvision.transforms as T
from hubconf import *
from util.misc import nested_tensor_from_tensor_list

torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO classes
CLASSES = [
    'fish', 'jellyfish', 'penguin','puffin','shark','starfish','stingray'
]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.00001

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    anImg = transform(im)
    data = nested_tensor_from_tensor_list([anImg])

    # propagate through the model
    outputs = model(data)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

    keep = probas.max(-1).values > 0.7  # 0.7 好像是调整置信度的
    # print(probas[keep])
    
    # get the label indices
    predicted_labels_idx = probas[keep].argmax(-1)

    # convert label indices to actual labels
    # predicted_labels = [CLASSES[i] for i in predicted_labels_idx]
    
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    
    return probas[keep], bboxes_scaled , predicted_labels_idx


if __name__ == "__main__":
    
    model = detr_resnet50(False, 7) # 这里与前面的num_classes数值相同，就是最大的category id值 + 1
    state_dict = torch.load("outputs/checkpoint.pth", map_location='cpu')
    # detr-r50_8.pth 還沒有 checkpoint 先試試看
    
    # state_dict = torch.load("detr-r50_7.pth", map_location='cpu')
    model.load_state_dict(state_dict["model"])
    model.eval()

    # im = Image.open('data/coco/train2017/001554.jpg')
    # im = Image.open(r'F:\A_Publicdatasets\RDD2022_released_through_CRDDC2022\RDD2022\A_unitedataset\images\val\China_Drone_000038.jpg')

    # scores, boxes = predict(im, model, transform)
    
    results = {}
    device = 'cuda'
    with torch.no_grad():
        image_folder_path = 'hw1_dataset/test2017'

        for filename in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, filename)
            im = Image.open(image_path)

            scores, boxes , id = predict(im, model, transform)
            
            # Store results in the desired format
            results[filename] = {"boxes": boxes.tolist(), "labels" : id.tolist() , "scores": scores.tolist()}
    
    with open("predictions.json", "w") as file:
        json.dump(results, file)



