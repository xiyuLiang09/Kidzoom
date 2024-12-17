import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.torch_utils import select_device


def get_keypoints(source: str):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()
    
    if torch.cuda.is_available():
        model.half().to(device)

    image = cv2.imread(source)
    image = letterbox(image, 960, stride=64, auto=True)[0]
    # image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)
    output, _ = model(image)

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    output = np.around(output, decimals=2)
    # print(output.tolist())
    # results = []
    # keypoints = []
    # for idx in range(output.shape[0]):
    #     keypoints = [output[0][i:i+2] for i in range(7, len(output[0]), 3)]
    #     results.append(keypoints)

    return output


def rescale_output(img1_shape, output, img0_shape, ratio_pad=None):
    
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    # rescale boxes
    output[:, [2, 4]] -= pad[0]  # x padding
    output[:, [3, 5]] -= pad[1]  # y padding
    output[:, 2:6] /= gain
    clip_boxes(output[:, 2:6], img0_shape)

    # rescale keypoints
    print(output.shape)
    output[:, range(7, len(output[0]), 3)] -= pad[0] # x padding
    # print(output[:, range(7, len(output), 3)])
    output[:, range(7, len(output[0]), 3)] /= gain
    output[:, range(8, len(output[0]), 3)] -= pad[1] # y padding
    output[:, range(8, len(output[0]), 3)] /= gain

    return output
    
    
def clip_boxes(boxes, img_shape):
    np.clip(boxes[:, 0], 0, img_shape[1])
    np.clip(boxes[:, 1], 0, img_shape[0])
    np.clip(boxes[:, 2], 0, img_shape[1])
    np.clip(boxes[:, 3], 0, img_shape[0])
