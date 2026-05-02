from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_transforms(image_size=(256, 256), augment=True):

    base_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ]

    image_transforms = base_transforms + [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    mask_transforms = base_transforms.copy()
    
    if augment:
        aug_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        
        image_transforms = [transforms.Resize(image_size)] + aug_transforms + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        mask_transforms = [transforms.Resize(image_size)] + aug_transforms[:2] + [
            transforms.ToTensor()
        ]
    return transforms.Compose(image_transforms), transforms.Compose(mask_transforms)
