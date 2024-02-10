import numpy as np
import math


optimizer = "adamW"

# Original values
orig_batch_size = 4*8
orig_lr = 0.001
orig_beta1 = 0.9
orig_beta2 = 0.999

# Newvalues
new_batch_size = 4
new_lr = 0
new_beta1 = 0
new_beta2 = 0

ratio = new_batch_size / orig_batch_size
if optimizer in ["adam", "adamW"] :
    new_lr = orig_lr * math.sqrt(ratio)
    new_beta1 = 1 - (ratio * (1 - orig_beta1))
    new_beta2 = 1 - (ratio * (1 - orig_beta2))
else :
    new_lr = orig_lr * ratio
print(ratio, new_lr, new_beta1, new_beta2)