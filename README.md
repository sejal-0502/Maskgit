# Next Frame MaskGIT

We provide the initial version of the Next Frame MaskGIT model for autoregressive next frame prediction.

## Repository Structure
To use both tokenizer and maskgit repositories together, place them in the same parent directory. This allows the MaskGIT repository to refer to the tokenizer files. The tokenizer is referred as ```visual_tokenization``` in this repo.

### Using `pytorch-lightning`
The training script `main_pl.py` uses `omegaconf` and lightning to train the model. Example:
```
python main_pl.py --base configs/maskgit.yaml -t True --n_gpus=1
```
