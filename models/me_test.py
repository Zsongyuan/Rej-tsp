import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
import MinkowskiEngine as ME

points = [torch.tensor([[1,1,0,0,0,0]])]
coordinates, features = ME.utils.batch_sparse_collate(
        [(p[:, :3], p[:, 3:] if p.shape[1] > 3 else p[:, :3]) for p in points],
        device=points[0].device)
print(coordinates,features) 