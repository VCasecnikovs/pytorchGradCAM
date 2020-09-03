# pytorchGradCAM
Pytorch gradcam implementation. Easy to use for the basic and seamese networks.

## Example how to use
Instantiate:
```
from gradCAM import GradCam
m = some_model()
chosen_module = m.encoder.layer3.conv1
gc = GradCam(m, chosen_module)
```

Get gradCAM
```
#All inputs for model and index of chosen class
#If index is None -> will be provided attention to the most chosen class
att_shot = gc(model_input1, model_input2, index=10)
```

Get beautiful heatmap
```
#Get heatmap on image and heatmap
image_with_gcam, heatmap = gc.mask_on_image(x[0],  att_shot[0], alpha = 0.4)
```
