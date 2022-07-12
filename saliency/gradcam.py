import os

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from utils.model import baseline


def get_grad_cam(model, target_layer, image_path, target_category=None):
    """

    :param model: Model to perform gradcam on
    :param target_layer: target layer in the model
    :param image_path: rgb image path
    :param target_category: class of target
    :return: cam
    """
    out = os.path.join("output", "saliency")
    os.makedirs(out, exist_ok=True)
    image_name = image_path.split(os.sep)[-1][:-4]
    rgb_image = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_image = np.float32(rgb_image) / 255
    input_tensor = preprocess_image(rgb_image,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    with GradCAM(model=model, target_layer=target_layer, use_cuda=False) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out, image_name + "_saliency.jpg"), cam_image)


if __name__ == "__main__":
    ## Path to img folder
    path = "data/Intersection/Front_Video/Day/Front_Video_Images "
    net = baseline(name="resnext50")
    weight = "output/baseline/resnext50/weights/LR_0.001_OPT_ADAM_BATCH_32_SHAPE_(240, 360)/LR_0.001_OPT_ADAM_BATCH_32_SHAPE_(240, 360)_best_weight.pth"
    net.load_state_dict(torch.load(weight, map_location="cpu"))
    target = list(net.cnn.children())[-3][-1]
    for idx, img in enumerate(os.listdir(path)):
        img_path = os.path.join(path,img)
        while idx < 10 :
            get_grad_cam(model=net, target_layer=target, image_path=img_path)
