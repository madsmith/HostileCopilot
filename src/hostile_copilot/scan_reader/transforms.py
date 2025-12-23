from PIL import Image
import torch
from torchvision import transforms

def get_crnn_transform(height=32, width=175):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        ResizeToHeight(height),
        transforms.ToTensor(),              # PIL → [C,H,W] float32
        transforms.Normalize((0.5,), (0.5,)),
        PadToWidth(width),
    ])

class ResizeToHeight:
    def __init__(self, height: int, interpolation=Image.BILINEAR):
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Resize image to given height while preserving aspect ratio.
        """
        width, height = img.size
        new_height = self.height
        new_width = int(round(width * (new_height / height)))
        return img.resize((new_width, new_height), self.interpolation)
    
    def __repr__(self):
        return f"ResizeToHeight(height={self.height})"

class PadToWidth:
    def __init__(self, width: int):
        self.width = width

    def __call__(self, img: torch.Tensor):
        # img: (1, H, W)
        channels, height, width = img.shape

        if width >= self.width:
            return img[:, :, :self.width]

        # How many pixels to add
        pad_amount = self.width - width

        # Rightmost column (C, H, 1)
        last_col = img[:, :, -1:].clone()

        # Repeat it pad_amount times → (C, H, pad_amount)
        pad = last_col.repeat(1, 1, pad_amount)

        # Concatenate
        return torch.cat([img, pad], dim=2)
    
    def __repr__(self):
        return f"PadToWidth(width={self.width})"
