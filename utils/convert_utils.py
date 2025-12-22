import torch

def onehot_to_greyscale(onehot_pred: torch.Tensor) -> torch.Tensor:
    """
    Converts a one-hot encoded prediction tensor to a grayscale mask.

    Args:
        onehot_pred (torch.Tensor): One-hot encoded prediction tensor of shape (C, H, W), 
            where C is the number of classes, H is the height, and W is the width. 
            The values are expected to be softmax probabilities.
    Returns:
        composite_array (numpy): Grayscale mask tensor of shape (H, W), 
            where each pixel value corresponds to the predicted class, 
            scaled to the range [0, 255].
    """
    assert onehot_pred.ndim == 3, f"Mismatch: expect input in shape (C, H, W), got {onehot_pred.shape}"
    channels, height, width = onehot_pred.shape
    composite_array = torch.zeros((height, width), dtype=torch.uint8)               # (H, W)
    pred_classes = torch.argmax(onehot_pred, dim=0)                                 # (H, W)
    composite_array = (pred_classes * 255 / (channels - 1)).round().to(torch.uint8).cpu().numpy() # (H, W)
    return composite_array

def neg1pos1_to_greyscale(img: torch.Tensor) -> torch.Tensor:
    """
    Converts a single-channel image tensor in the range [-1, 1] to a grayscale image in the range [0, 255].

    Args:
        img (torch.Tensor): Image tensor of shape (H, W) with values in [-1, 1].
    Returns:
        img (numpy.ndarray): Grayscale image of shape (H, W) with values in [0, 255].
    """
    if img.ndim == 3:
        img = img[:,:,0]
    assert img.ndim == 2, f"Mismatch: expect input in shape (H, W), got {img.shape}"
    img = (img + 1.) / 2. * 255.
    img = img.cpu().numpy()
    return img

def softmax_to_greyscale(probs: torch.Tensor) -> torch.Tensor:
    """
    Converts a per-channel softmax tensor to a grayscale image in the range [0, 255].

    Args:
        probs (torch.Tensor): Per-channel softmax probabilities of shape (C, H, W).
    Returns:
        img (numpy.ndarray): Grayscale image of shape (H, W) with values in [0, 255].
    """
    assert probs.ndim == 3, f"Mismatch: expect input in shape (C, H, W), got {probs.shape}"
    channels, height, width = probs.shape
    composite_array = torch.zeros((height, width), dtype=torch.uint8)
    pred_classes = torch.argmax(probs, dim=0)
    composite_array = (pred_classes * 255 / (channels - 1)).round().to(torch.uint8).cpu().numpy()
    return composite_array