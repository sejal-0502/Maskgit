import random
import math
import torch
from tqdm import tqdm
import numpy as np
import pytorch_lightning as pl


# TODO template?

class UnconditionalMaskGITScheduler(pl.LightningModule):
    def __init__(self, *, num_tokens, mask_value, codebook_size, default_schedule_mode_train="arccos", default_schedule_mode="arccos", default_num_steps=12, disable_bar=False) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.mask_value = mask_value
        self.default_schedule_mode = default_schedule_mode
        self.default_schedule_mode_train = default_schedule_mode_train
        self.default_num_steps = default_num_steps
        self.disable_bar = disable_bar

    def get_mask_code(self, code, mode=None, value=None):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        mode = mode or self.default_schedule_mode_train
        batch_size, seq_len = code.shape
        r = torch.rand(code.size(0))
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        elif mode == 'cubic':
            val_to_mask = 1 - (r ** 3)
        elif mode == 'pow4':
            val_to_mask = 1 - (r ** 4)
        elif mode == 'pow6':
            val_to_mask = 1 - (r ** 6)
        elif mode == 'arccos2':
            val_to_mask = torch.arccos(r**2) / (math.pi * 0.5)
        else:
            val_to_mask = None

        mask_code = code.detach().clone()
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(batch_size, 1)  # mask tensor with the same shape as code

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a random token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, self.codebook_size)

        return mask_code, mask
    
    def get_fixed_mask_code(self, code, mask_ratio=0.5, value=None):
        """ Replace the code token by *mask_value* according to the *mask* tensor
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mask  -> torch.BoolTensor(): bsize * 16 * 16, the binary mask of the mask
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
        """
        batch_size, seq_len = code.shape
        num_masked_tokens = int(seq_len * mask_ratio)

        masked_code = code.clone()
        mask = torch.zeros_like(code, dtype=torch.bool)

        for i in range(batch_size):
            # Randomly select indices to mask
            indices = torch.randperm(seq_len)[:num_masked_tokens]
            mask[i, indices] = True

        if value is not None:
            masked_code[mask] = value
        else:
            masked_code[mask] = torch.randint(0, self.codebook_size, (mask.sum(),), device=code.device)

        return masked_code, mask

    def adap_sche(self, num_steps=None, schedule_mode=None, leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        num_steps = num_steps or self.default_num_steps
        schedule_mode = schedule_mode or self.default_schedule_mode
        r = torch.linspace(1, 0, num_steps)
        if schedule_mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif schedule_mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif schedule_mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif schedule_mode == "cubic":
            val_to_mask = 1 - (r ** 3)
        elif schedule_mode == "pow4":
            val_to_mask = 1 - (r ** 4)
        elif schedule_mode == "pow6":
            val_to_mask = 1 - (r ** 6)
        elif schedule_mode == "inv_root":
            val_to_mask = (r ** .5)
        elif schedule_mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif schedule_mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        elif schedule_mode == "arccos2":
            val_to_mask = torch.arccos(r**2) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.num_tokens * self.num_tokens)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.num_tokens * self.num_tokens) - sche.sum()         # need to sum up nb of code

        return tqdm(sche.int(), leave=leave, disable=self.disable_bar)
