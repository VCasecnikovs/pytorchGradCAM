import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List

class GradCam():

    output_activations = []
    output_gradients = []

    def forward_hook(self, module, input, output):
        self.output_activations.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        self.output_gradients.append(grad_output[0])

    def __init__(self, model : torch.nn.Module, module : torch.nn.Module):
        self.model = model

        module.register_forward_hook(self.forward_hook)
        module.register_backward_hook(self.backward_hook)

    def forward(self, input):
        return self.model(input)

    def __call__(self, *input : Tuple[torch.Tensor], index=None) -> List[torch.Tensor]:
        self.output_activations : List[torch.Tensor] = []
        self.output_gradients : List[torch.Tensor] = []

        output = self.model(*input)

        if index == None:
            index = torch.argmax(output)

        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32) #Делаем one_hot матрицу оставляя только любимый класс
        one_hot[0, index] = 1
        criterion = (one_hot * output).sum()
        self.model.zero_grad()
        criterion.backward()

        #We are assuming that if we have one image as an input, we should produce all cams to it. But if we have more than one. Than each activation should be for the specific image.
        cams = []
        for i in range(len(self.output_gradients)):
            activation = self.output_activations[i]
            gradient = self.output_gradients[i]
            grad_mean = gradient.mean((2,3))
            gmh = grad_mean.shape
            grad_mean = grad_mean.view(gmh[0], gmh[1], 1, 1)
            cam = activation * grad_mean
            cam = cam.sum(1, keepdim=True)
            cam, _ = torch.max(cam, 0)
            cam = (cam - cam.min())/(cam.max() - cam.min())
            cams.append(cam)

        return cams
    
    def mask_on_image(self, image : torch.Tensor, mask : torch.Tensor, alpha = 0.4):
        mask = mask.detach()
        image = image.detach()
        mask = torch.nn.functional.interpolate(mask[None], size=image.shape[-2:])[0]
        heatmap = cv2.applyColorMap((mask * 255).permute(1, 2, 0).numpy().astype(np.uint8), cv2.COLORMAP_JET)
        image = (image * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        cam = image + (heatmap * alpha).astype(np.uint8)
        return cam, heatmap





