# Trainer for MaskGIT
import os
import math

import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import pytorch_lightning as pl

# from Trainer.trainer import Trainer
from final_titok.MastersProject_TiTok.taming.util import instantiate_from_config


class MaskGIT(pl.LightningModule):

    def __init__(self, *, tokenizer_config, predictor_config, loss_config, scheduler_config, num_tokens=64, patch_size=16, image_size=256, grad_acc_steps=1, adjust_lr_to_batch_size=False, load_tokenizer_checkpoint=True):
        """ Initialization of the model (VQGAN and Masked Transformer), optimizer, criterion, etc."""
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.grad_acc_steps = grad_acc_steps
        self.adjust_lr_to_batch_size = adjust_lr_to_batch_size

        # Tokenizer
        self.ae = self.build_tokenizer(tokenizer_config, load_tokenizer_checkpoint)
        # self.num_tokens = self.image_size // self.patch_size # change may be needed here
        self.num_tokens = num_tokens
        print("Number of tokens:", self.num_tokens)

        # Predictor
        self.vit = self.build_predictor(predictor_config)
        self.codebook_size = self.vit.codebook_size
        self.mask_value = self.codebook_size

        # Loss and Optimizer
        self.criterion = instantiate_from_config(loss_config)

        # Sampler
        for field in ["codebook_size", "num_tokens", "mask_value"]:
            scheduler_config.params.setdefault(field, getattr(self, field))
        self.scheduler = instantiate_from_config(scheduler_config)
        
        self.window_loss = []

        self.similarity_matrix = self.build_similarity_matrix()

    def train(self, mode=True):
        # Override the train method
        super().train(mode)  # set the rest of the model to train/eval mode
        self.ae.eval()  # ensure the submodule is always in eval mode

    def eval(self):
        # Override the eval method
        super().eval()  # set the rest of the model to eval mode
        self.ae.eval()  # ensure the submodule is always in eval mode

    # Respobsible for loading the checkpoints 
    def build_tokenizer(self, tokenizer_config, load_checkpoint=True):
        tokenizer_folder = os.path.expandvars(tokenizer_config.folder)
        checkpoint_folder = os.path.expandvars(tokenizer_config.checkpoint_folder)
        # Load config and create
        tokenizer_config = OmegaConf.load(os.path.join(tokenizer_folder, "config.yaml"))
        model = instantiate_from_config(tokenizer_config.model)
        # Load checkpoint, we only load it when training, otherwise it's stored in the main model checkpoint
        if load_checkpoint:
            checkpoint = torch.load(os.path.join(tokenizer_folder, checkpoint_folder), map_location="cpu")["state_dict"]
            # delete all dino keys from the checkpoint
            dk = [k for k in checkpoint.keys() if "dino" in k]
            _ = [checkpoint.pop(k) for k in dk]
            o = model.load_state_dict(checkpoint, strict=False)
            if not (all(["dino" in k for k in o.unexpected_keys]) and all(["dino" in k for k in o.missing_keys])):
                model.load_state_dict(checkpoint, strict=False) # this will raise an error if the model is not the same except for dino params
        model = model.eval()

        try:
            self.image_size = tokenizer_config.model.params.encoder_config.params.image_size
            self.patch_size = tokenizer_config.model.params.encoder_config.params.patch_size
        except:
            try:
                assert (model.image_width == model.image_height) and (model.encoder.patch_height == model.encoder.patch_width)
                self.image_size = model.encoder.image_height
                self.patch_size = model.encoder.patch_height
            except:
                print("Could not infer image size and patch size from tokenizer model, using default values")

        return model

    def build_predictor(self, predictor_config):
        # merge the self.config.model.predictor config with self.codebook_size, if not already present
        if "codebook_size" not in predictor_config.params:
            predictor_config.params.codebook_size = self.ae.quantize.n_e
        model = instantiate_from_config(predictor_config)
        return model

    def configure_learning_rate(self, base_lr, batch_size, num_gpus):
        if self.adjust_lr_to_batch_size:
            # Set the learning rate according to batch size, number of GPUs and gradient accumulation steps
            bs_acc_factor = max(self.grad_acc_steps//8, 1)
            self.learning_rate =  num_gpus * batch_size * base_lr * bs_acc_factor
            print("Setting learning rate to {:.2e} = {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr) * {} (grad accumulation factor)".format(
                self.learning_rate, num_gpus, batch_size, base_lr, bs_acc_factor))
        else:
            self.learning_rate = base_lr
            print("Setting learning rate to {:.2e} (base learning rate)".format(self.learning_rate))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.vit.parameters(), lr=self.learning_rate, weight_decay=1e-5, betas=(0.9, 0.96))
        return optimizer

    def get_input(self, batch, k):
        if isinstance(batch, dict):
            x = batch[k]
            print("Shape of X in get_input in if:", x.shape)
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format) # [b, c, h, w]
        else:
            x = batch
            print("Shape of X in get_input in else:", x.shape) # [8, 3, 256, 256]
            # x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN
        return x.float()

    def build_similarity_matrix(self):
        embeddings = self.ae.quantize.embedding.weight.data
        print("Embedding shape:", embeddings.shape)
        # Compute the distance matrix
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
        #similarity_matrix = torch.cdist(norm_embeddings, norm_embeddings, p=2)
        #torch.fill_diagonal_(similarity_matrix, 0)
        
        # calculate row norm of similarity matrix
        row_norms = torch.norm(similarity_matrix, p=2, dim=1, keepdim=True)
        normalized_similarity_matrix = similarity_matrix / row_norms
        return normalized_similarity_matrix.to(self.device)

    def get_soft_codes(self, gt_codes, similarity_matrix, temperature=1.0):
        soft_code_vectors = similarity_matrix[gt_codes.view(-1)].view(gt_codes.size(0), gt_codes.size(1)*gt_codes.size(2), -1)
        soft_code_vectors = F.softmax(soft_code_vectors / temperature, dim=-1)
        zeros_mat = torch.zeros(soft_code_vectors.size(0), soft_code_vectors.size(1), 1).to(self.device)
        soft_code_vectors = torch.cat([soft_code_vectors, zeros_mat], dim=-1)
        return soft_code_vectors

    @staticmethod
    def get_mask_code(code, mode="arccos", value=None, codebook_size=1024):
        """ Replace the code token by *value* according the the *mode* scheduler
           :param
            code  -> torch.LongTensor(): bsize * 16 * 16, the unmasked code
            mode  -> str:                the rate of value to mask
            value -> int:                mask the code by the value
           :return
            masked_code -> torch.LongTensor(): bsize * 16 * 16, the masked version of the code
            mask        -> torch.LongTensor(): bsize * 16 * 16, the binary mask of the mask
        """
        #print (f'current max ratio {max_masking_ratio}')
        batch_size, seq_len = code.shape
        # r = torch.rand(code.size(0)) # generate random float for each sample in the batch
        r = torch.rand(batch_size) # generate random float for each sample in the batch
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = (torch.arccos(r) / (math.pi * 0.5)) #*max_masking_ratio
        else:
            val_to_mask = None

        mask_code = code.detach().clone() # clones the code tensor to avoid in-place operations
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(batch_size, seq_len) < val_to_mask.view(batch_size, 1) # mask tensor with the same shape as code

        if value > 0:  # Mask the selected token by the value
            mask_code[mask] = torch.full_like(mask_code[mask], value)
        else:  # Replace by a randon token
            mask_code[mask] = torch.randint_like(mask_code[mask], 0, codebook_size) # replace masked tokens with random values from 0 to codebook_size

        return mask_code, mask

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, 'image')
        print("Shape of input x:", x.shape) # [8, 3, 256, 256]
        # x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

        # VQGAN encoding to img tokens
        with torch.no_grad():
            emb, _, [_, _, code] = self.ae.encode(x) # your input x shape matters here, code is the quantized code indices
            # emb, middle, last = self.ae.encode(x)
            print("emb shape:", emb.shape)
            # print("middle:", middle)          # inspect or print shape if tensor
            # print("last:", last)              # last is a list [_, _, code]
            # print("last[0]:", last[0])
            # print("last[1]:", last[1])
            print("Code shape:", code.shape)
            # print("Encoder output shape : ", self.ae.encode(x)) # [8, 16, 16] for 256x256 images with patch size 16
            # print("code shape before reshape:", code.shape)
            # print("code numel (total elements):", code.numel())

            # code = code.reshape(x.size(0), self.num_tokens, self.num_tokens)
            code = code.reshape(x.size(0), self.num_tokens)
            print("Code shape 2:", code.shape)
            print("Code size:", code.size())

        # Mask the encoded tokens
        masked_code, mask = self.scheduler.get_mask_code(code, value=self.mask_value)
        # masked_code, mask = self.get_mask_code(code, value=self.mask_value, codebook_size=self.codebook_size)
        
        pred = self.vit(masked_code)  # The unmasked tokens prediction
        # Cross-entropy loss
        loss = self.criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1)) / self.grad_acc_steps

        self.window_loss.append(loss.data.cpu().numpy().mean())

        # update weight if accumulation of gradient is done
        update_grad = batch_idx % self.grad_acc_steps == self.grad_acc_steps - 1
        if update_grad:
            self.log("train/ce_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/ce_loss_window", np.array(self.window_loss).sum(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.window_loss = []

        return loss

    def log_images(self, batch, **kwargs):
        x = self.get_input(batch, 'image')
        # x = 2 * x - 1  # normalize from x in [0,1] to [-1,1] for VQGAN

        # VQGAN encoding to img tokens
        with torch.no_grad():
            emb, _, [_, _, code] = self.ae.encode(x)
            print("emb shape img logs:", emb.shape)
            print("code shape img logs:", code.shape)
            # code = code.reshape(x.size(0), self.num_tokens, self.num_tokens)
            code = code.reshape(x.size(0), self.num_tokens)
            # decode
            ae_recon = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
            # ae_recon = self.ae.decode(torch.clamp(code, 0, self.codebook_size-1))
            ae_recon = ae_recon[0] if isinstance(ae_recon, tuple) else ae_recon
            ae_recon = (ae_recon + 1.0) / 2.0

        # Mask the encoded tokens
        masked_code, mask = self.scheduler.get_mask_code(code, value=self.mask_value)
        print("Mask shape in log_images:", mask.shape) # [8, 64]
        # masked_code, mask = self.get_mask_code(code, value=self.mask_value, codebook_size=self.codebook_size)
        
        pred = self.vit(masked_code)  # The unmasked tokens prediction

        # Generate sample for visualization
        # gen_sample = self.sample(num_samples=10)[0]
        gen_sample = self.sample(num_samples=x.size(0))[0]
        gen_sample = gen_sample[0] if isinstance(gen_sample, tuple) else gen_sample

        gen_sample = vutils.make_grid(gen_sample, nrow=10, padding=2, normalize=False)
        
        # Show reconstruction
        unmasked_code = torch.softmax(pred, -1).max(-1)[1] # [8, 64]
        N = min(10, x.size(0))
        reco_sample = self.reconstruct(x=x[:N], code=code[:N], unmasked_code=unmasked_code[:N], mask=mask[:N])
        # Add the AE reconstruction 
        chunks = torch.chunk(reco_sample, 3, dim=0)
        reco_sample = torch.cat([chunks[0], ae_recon[:N], *chunks[1:]], 0)
        # reco_sample = vutils.make_grid(reco_sample, nrow=N, padding=2, normalize=False)
        reco_sample = vutils.make_grid(reco_sample.data, nrow=N, padding=2, normalize=False)

        ret = {"sampled": gen_sample, "reconstructed": reco_sample}
        # unnormalize
        # ret = {k: (v + 1.0) / 2.0 for k, v in ret.items()}
        return ret


    def reconstruct(self, x=None, code=None, masked_code=None, unmasked_code=None, mask=None):
        """ For visualization, show the model ability to reconstruct masked img
           :param
            x             -> torch.FloatTensor: bsize x 3 x 256 x 256, the real image
            code          -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
            masked_code   -> torch.LongTensor: bsize x 16 x 16, the masked image tokens
            unmasked_code -> torch.LongTensor: bsize x 16 x 16, the prediction of the transformer
            mask          -> torch.LongTensor: bsize x 16 x 16, the binary mask of the encoded image
           :return
            l_visual      -> torch.LongTensor: bsize x 3 x (256 x ?) x 256, the visualization of the images
        """

        l_visual = [x]
        with torch.no_grad():
            if code is not None:
                # code = code.view(code.size(0), self.num_tokens, self.num_tokens) 
                code = code.view(code.size(0), self.num_tokens) 
                # Decoding reel code
                _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1)) # full resolution images
                print("Shape of _x in reconstruct:", _x.shape) # [8, 3, 256, 256]
                _x = _x[0] if isinstance(_x, tuple) else _x  # if dino loss is included
                if mask is not None:
                #     # Decoding reel code with mask to hide
                #     # mask = mask.view(code.size(0), 1, self.num_tokens, self.num_tokens).float()
                #     mask = mask.view(code.size(0), self.num_tokens).float()
                #     print("Mask shape in reconstruct:", mask.shape)
                #     __x2 = _x * (1 - F.interpolate(mask, (self.image_size, self.image_size)).to(self.device))
                #     l_visual.append(__x2)
                    print("Mask shape for visualization:", mask.shape) 
                    B, T = mask.shape
                    side = int(math.sqrt(T))
                    assert side * side == T, "Mask shape is not a square"
                    mask_grid = mask.view(B, 1, side, side).float()  
                    mask_full = F.interpolate(mask_grid, size=(self.image_size, self.image_size), mode='nearest')
                    __x2 = _x * (1.0 - mask_full.to(self.device))
                    l_visual.append(__x2)
                
            if masked_code is not None:
                # Decoding masked code
                # masked_code = masked_code.view(code.size(0), self.num_tokens, self.num_tokens)
                masked_code = masked_code.view(code.size(0), self.num_tokens)
                __x = self.ae.decode_code(torch.clamp(masked_code, 0, self.codebook_size-1))
                __x = __x[0] if isinstance(__x, tuple) else __x  # if dino loss is included
                l_visual.append(__x)

            if unmasked_code is not None:
                # Decoding predicted code
                # unmasked_code = unmasked_code.view(code.size(0), self.num_tokens, self.num_tokens)
                unmasked_code = unmasked_code.view(code.size(0), self.num_tokens)
                ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
                ___x = ___x[0] if isinstance(___x, tuple) else ___x # if dino loss is included
                l_visual.append(___x)

        return (torch.cat(l_visual, dim=0) + 1.0) / 2.0

    
    def sample(self, init_code=None, num_samples=50, labels=None, sm_temp=1, w=0, randomize="linear", r_temp=4.5, schedule_mode=None, num_steps=None):
        """ Generate sample with the MaskGIT model
           :param
            init_code       -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            num_samples     -> int:              the number of images to generate
            sm_temp         -> float:            the temperature before softmax
            randomize       -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp          -> float:            temperature for the randomness
            schedule_mode   -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            num_steps:      -> int:              number of step for the decoding
           :return
            x               -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code            -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            drop = torch.ones(num_samples, dtype=torch.bool).to(self.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == self.codebook_size).float().view(num_samples, self.num_tokens*self.num_tokens)
            else:  # Initialize a code
                if self.mask_value < 0:  # Code initialize with random tokens
                    # code = torch.randint(0, self.codebook_size, (num_samples, self.num_tokens, self.num_tokens)).to(self.device)
                    code = torch.randint(0, self.codebook_size, (num_samples, self.num_tokens)).to(self.device)
                else:  # Code initialize with masked tokens
                    # code = torch.full((num_samples, self.num_tokens, self.num_tokens), self.mask_value).to(self.device)
                    code = torch.full((num_samples, self.num_tokens), self.mask_value).to(self.device)
                # mask = torch.ones(num_samples, self.num_tokens*self.num_tokens).to(self.device)
                mask = torch.ones(num_samples, self.num_tokens).to(self.device)

            # Instantiate scheduler
            scheduler = self.scheduler.adap_sche(num_steps, schedule_mode)

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                if w != 0:
                    # Model Prediction
                    logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                        torch.cat([labels, labels], dim=0),
                                        torch.cat([~drop, drop], dim=0))
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                    _w = w * (indice / (len(scheduler)-1))
                    # Classifier Free Guidance
                    logit = (1 + _w) * logit_c - _w * logit_u
                else:
                    logit = self.vit(code.clone())

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                # conf = torch.gather(prob, 2, pred_code.view(num_samples, self.num_tokens*self.num_tokens, 1))
                conf = torch.gather(prob, 2, pred_code.view(num_samples, self.num_tokens, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    # rand = r_temp * np.random.gumbel(size=(num_samples, self.num_tokens*self.num_tokens)) * (1 - ratio)
                    rand = r_temp * np.random.gumbel(size=(num_samples, self.num_tokens)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(num_samples, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                # conf = (conf >= tresh_conf.unsqueeze(-1)).view(num_samples, self.num_tokens, self.num_tokens)
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(num_samples, self.num_tokens)
                # f_mask = (mask.view(num_samples, self.num_tokens, self.num_tokens).float() * conf.view(num_samples, self.num_tokens, self.num_tokens).float()).bool()
                f_mask = (mask.view(num_samples, self.num_tokens).float() * conf.view(num_samples, self.num_tokens).float()).bool()
                code[f_mask] = pred_code.view(num_samples, self.num_tokens)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                # l_codes.append(pred_code.view(num_samples, self.num_tokens, self.num_tokens).clone())
                # l_mask.append(mask.view(num_samples, self.num_tokens, self.num_tokens).clone())
                l_codes.append(pred_code.view(num_samples, self.num_tokens).clone())
                l_mask.append(mask.view(num_samples, self.num_tokens).clone())

            # decode the final prediction
            code = torch.clamp(code, 0, self.codebook_size-1)
            x = self.ae.decode_code(code)

        x = x[0] if isinstance(x, tuple) else x
        x = (x + 1.0) / 2.0
        self.vit.train()
        return x, l_codes, l_mask


