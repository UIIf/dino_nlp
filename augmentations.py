from typing import Literal, Union
import numpy as np
import torch


class MulticropAugmentations(object):

    def __init__(
        self,
        tokenizer,
        max_token_count: int,
        pad_token_id:int,
        return_mask:bool=True,
        *,
        crop_sep_method: Literal["Tokens", "Lines"] = "Tokens",
        global_crop_ratio: Union[float, tuple[float, float]] = (0.6, 0.8),
        local_crop_ratio: Union[float, tuple[float, float]] = (0.3, 0.4),
        global_crop_count: int = 2,
        local_crop_count: int = 4,
    ):
        """Generate multi crop augmentations
Returns: (
        [global_crop_count, gloabl_tokens_count],
        [local_crop_count, gloabl_tokens_count * local_crop_ratio]
)

Args:
        crop_sep_method: 
            Tokens: split code by tokens
            Lines: split code by lines, then tokenize

        global_crop_ratio:
            #TODO
"""
        match crop_sep_method:
            case "Tokens":
                self.crop_sep = _Separate_by_tokens(
                    tokenizer=tokenizer,
                    max_token_count=max_token_count,
                    pad_token_id=pad_token_id,
                )
            case "Lines":
                pass
            case _:
                raise ValueError("No matchin crop_sep_method")
        self.max_token_count = max_token_count
        self.crop_sep_method = crop_sep_method
        self.gloabl_tokens_ratio = global_crop_ratio
        self.local_crop_ratio = local_crop_ratio
        self.global_crop_count = global_crop_count
        self.local_crop_count = local_crop_count
        self.rng = np.random.default_rng()
        self.return_mask = return_mask

    def __call__(self, x):
        if isinstance(self.gloabl_tokens_ratio, float):
            global_counts = [self.gloabl_tokens_ratio] * self.global_crop_count
        else:
            global_counts = [
                self.rng.uniform(*self.gloabl_tokens_ratio)
                for i in range(self.global_crop_count)
            ]

        if isinstance(self.local_crop_ratio, float):
            local_counts = [self.local_crop_ratio] * self.local_crop_count
        else:
            local_counts = [
                self.rng.uniform(*self.local_crop_ratio)
                for i in range(self.local_crop_count)
            ]

        toks, mask = self.crop_sep(x, global_counts + local_counts)
        if self.return_mask:
            return toks, mask
        else:
            return toks 

class _Separate_by_tokens(object):
    
    def __init__(self, tokenizer, max_token_count, pad_token_id):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count
        self.pad_token_id = pad_token_id
        self.rng = np.random.default_rng()

    def __call__(self, x:str, counts:list[int]) ->list[list[int]]:
        tokenized = self.tokenizer(x, return_tensors="pt").input_ids[0]
        res = []
        masks = []     
        for i in counts:
            i = int(i * len(tokenized))
            offset = self.rng.integers(0, len(tokenized) - i)
            crop = tokenized[offset : offset + i][: self.max_token_count]
            res.append(
                torch.cat(
                    [
                        crop,
                        (torch.ones(self.max_token_count - len(crop)) * self.pad_token_id).to(torch.int32),
                    ]
                )
            )
            masks.append(torch.cat([torch.ones(len(crop)), torch.zeros(self.max_token_count - len(crop))]))

        return res, masks