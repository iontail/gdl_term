from .resnet import get_resnet
from .preactresnet import get_preactresnet

def get_model(model: str, num_classes: int, img_size: int):

    if img_size < 128:
        is_data_small = True
    else:
        is_data_small = False
    
    model = model.lower()
    if 'preactresnet' in model:
        model = get_preactresnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)

    elif 'resnet' in model:
        model = get_resnet(model_name=model, num_classes=num_classes, is_data_small=is_data_small)

    return model