import torch
import torchaudio.transforms as T
import numpy as np

def cancel_channel(samples: torch.Tensor|np.ndarray) -> torch.Tensor|np.ndarray:
    '''
    Args:
        samples: torch.Tensor|np.ndarray: The audio samples to cancel the channel, (channels, samples) or (samples, )
        
    Returns:
        torch.Tensor|np.ndarray: The audio samples with the channel canceled, (samples, )
    '''
    if isinstance(samples, torch.Tensor):
        match samples.ndim:
            case 1:
                return samples
            case 2:
                return samples.mean(dim=0, keepdim=False)
            case _:
                raise ValueError(f"Invalid samples shape: {samples.shape}")
    elif isinstance(samples, np.ndarray):
        match samples.ndim:
            case 1:
                return samples
            case 2:
                return samples.mean(axis=0, keepdims=False)
            case _:
                raise ValueError(f"Invalid samples shape: {samples.shape}")
    else:
        raise TypeError(f"Invalid samples type: {type(samples)}")

def add_channel(samples: torch.Tensor|np.ndarray) -> torch.Tensor|np.ndarray:
    '''
    Args:
        samples: torch.Tensor|np.ndarray: The audio samples to add the channel, (samples, ) or (channels, samples)
        
    Returns:
        torch.Tensor|np.ndarray: The audio samples with the channel added, (1, samples)
    '''
    if isinstance(samples, torch.Tensor):
        match samples.ndim:
            case 1:
                return samples.unsqueeze(0)
            case 2:
                return samples.mean(dim=0, keepdim=True)
            case _:
                raise ValueError(f"Invalid samples shape: {samples.shape}")
    elif isinstance(samples, np.ndarray):
        match samples.ndim:
            case 1:
                return samples[np.newaxis, :]
            case 2:
                return samples.mean(axis=0, keepdims=True)
            case _:
                raise ValueError(f"Invalid samples shape: {samples.shape}")
    else:
        raise TypeError(f"Invalid samples type: {type(samples)}")
    
def resample(samples: torch.Tensor|np.ndarray, src_rate: int, tgt_rate: int) -> torch.Tensor|np.ndarray:
    '''
    Args:
        samples: torch.Tensor|np.ndarray: The audio samples to preprocess, (channels, samples)
        src_rate: int: The source sample rate
        tgt_rate: int: The target sample rate
        
    Returns:
        torch.Tensor|np.ndarray: The preprocessed audio samples, (channels, samples)
    '''
    if src_rate == tgt_rate:
        return samples
    
    if isinstance(samples, np.ndarray):
        is_src_np = True
    # all to tensor
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples)
    
    # The T.Resample requires Double(float32)! Instead of float64
    samples = samples.to(dtype=torch.float32)
    samples = T.Resample(src_rate, tgt_rate)(samples)
    
    if is_src_np:
        samples = samples.numpy()
    return samples

def to_numpy(samples: torch.Tensor|np.ndarray) -> np.ndarray:
    '''
    Args:
        samples: torch.Tensor|np.ndarray: The audio samples to convert to numpy, (channels, samples)
        
    Returns:
        np.ndarray: The audio samples in numpy, (channels, samples)
    '''
    if isinstance(samples, torch.Tensor):
        return samples.numpy()
    elif isinstance(samples, np.ndarray):
        return samples
    else:
        raise TypeError(f"Invalid samples type: {type(samples)}")
    
def to_tensor(samples: torch.Tensor|np.ndarray) -> torch.Tensor:
    '''
    Args:
        samples: torch.Tensor|np.ndarray: The audio samples to convert to tensor, (channels, samples)
        
    Returns:
        torch.Tensor: The audio samples in tensor, (channels, samples)
    '''
    if isinstance(samples, torch.Tensor):
        return samples
    elif isinstance(samples, np.ndarray):
        return torch.tensor(samples)
    else:
        raise TypeError(f"Invalid samples type: {type(samples)}")