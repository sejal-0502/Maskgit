import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from einops import rearrange
from models.maskgit import MaskGIT


class MaskGITNextFrame(MaskGIT):
    def __init__(self, *, tokenizer_config, predictor_config, loss_config, scheduler_config, 
                 num_frames: int, drop_frame: float, drop_token: float, drop_token_type: str = 'mask',
                 grad_acc_steps=1, adjust_lr_to_batch_size=False, load_tokenizer_checkpoint=True,
                 input_data_format='bfchw', input_rescale=False):
        self.num_frames = num_frames
        self.drop_frame = drop_frame
        self.drop_token = drop_token
        self.drop_token_type = drop_token_type

        self.input_data_format = input_data_format
        self.input_rescale = input_rescale
        
        predictor_config.params.setdefault("num_frames", getattr(self, "num_frames"))

        super(MaskGITNextFrame, self).__init__(tokenizer_config=tokenizer_config, 
                                               predictor_config=predictor_config, 
                                               loss_config=loss_config, 
                                               scheduler_config=scheduler_config, 
                                               grad_acc_steps=grad_acc_steps,
                                               adjust_lr_to_batch_size=adjust_lr_to_batch_size, 
                                               load_tokenizer_checkpoint=load_tokenizer_checkpoint)
    
    def get_input(self, batch, k):
        assert len(batch.shape) == 5, 'input must be 5D tensor'
        x = batch
        
        if self.input_data_format == 'bcfhw':
            x = rearrange(x, 'b c f h w -> b f c h w')
        assert x.size(1) == self.num_frames, f'input must have {self.num_frames} frames'
        if self.input_rescale:
            raise RuntimeError('Rescaling should be done in the loader')
        return x

    def encode_frames(self, x):
        """
        Expects input of shape (batch, frames, channels, height, width)
        Returns:    the embeddings in the shape (batch, frames, channels, num_tokens, num_tokens)
                    the codes in the shape (batch, frames, num_tokens, num_tokens)
        """
        b, f, c, h, w = x.size()
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        emb, _, [_, _, code] = self.ae.encode(x)
        code = rearrange(code, '(b f h w) -> b f h w', b=b, f=f, h=self.num_tokens, w=self.num_tokens)
        emb = rearrange(emb, "(b f) c h w -> b f c h w", b=b, f=f, h=self.num_tokens, w=self.num_tokens)
        return emb, code

    def validation_step(self, batch, batch_idx):
        # load the image
        x = self.get_input(batch, 'image')
        N = x.size(0)
        
        x_0 = x[:, :-1]
        maskgit_output = self.sample(x_0=x_0[:N])
        gen_sample = maskgit_output["sampled_image"]
        logits_last = maskgit_output["logits"]

        if isinstance(gen_sample, tuple): # if dino loss is included
            gen_sample = gen_sample[0]
        
        # perplexity loss
        _, code = self.encode_frames(x)
        code_last = code[:, -1]
        logits_flat = logits_last.reshape(-1, self.codebook_size + 1)
        code_flat = code_last.reshape(-1)
        perplexity = self.criterion(logits_flat, code_flat).item()
        accuracy = (logits_flat.argmax(-1) == code_flat).float().mean().item()
        self.log("val/ce_loss", perplexity, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/accuracy", accuracy, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, 'image')

        # VQGAN encoding to img tokens
        with torch.no_grad():
            # encode all frames except the last one
            x_0 = x[:, :-1] 
            x_0 = x_0.reshape(-1, 3, self.image_size, self.image_size)
            _, _, [_, _, code_0] = self.ae.encode(x_0)
            code_0 = code_0.reshape(x_0.size(0), self.num_tokens, self.num_tokens) # [bsize*num_frames-1 x 16 x 16]

            # encode the last frame
            x_last = x[:, -1]
            _, _, [_, _, code_last] = self.ae.encode(x_last)
            code_last = code_last.reshape(x_last.size(0), self.num_tokens, self.num_tokens) # [bsize x 16 x 16]

        # drop complete temporal -- Randomly mask out the code_0
        drop_frame = torch.empty(code_0.size(0)).uniform_() < self.drop_frame
        code_0[drop_frame] = self.codebook_size

        # drop random token -- Randomly mask out the previous frames codes
        drop_token = torch.empty(code_0.size(0), self.num_tokens, self.num_tokens).uniform_() < self.drop_token
        if self.drop_token_type == 'mask':
            code_0[drop_token] = self.codebook_size
        elif self.drop_token_type == 'random':
            code_0_random = torch.randint(0, self.codebook_size, code_0.size()).to(self.device)
            code_0[drop_token] = code_0_random[drop_token]

        # Mask the last frame
        drop_token_last = torch.empty(code_last.size(0), self.num_tokens, self.num_tokens).uniform_() < self.drop_token
        if self.drop_token_type == 'random':
            code_last_random = torch.randint(0, self.codebook_size, code_last.size()).to(self.device)
            code_last[drop_token_last] = code_last_random[drop_token_last]
        masked_code_last, mask_last = self.scheduler.get_mask_code(code_last, value=self.mask_value)

        # Concatenate the masked code and the drop label
        combined_code = torch.cat([code_0.view(x.size(0), -1, self.num_tokens, self.num_tokens), 
                                   masked_code_last.view(x.size(0), 1, self.num_tokens, self.num_tokens)], 
                                   dim=1)
        
        pred = self.vit(combined_code, drop_frame=drop_frame, drop_token=drop_token)  # The unmasked tokens prediction
        # Cross-entropy loss
        pred_last = pred[:, -self.num_tokens*self.num_tokens:]
        loss_pred_last = self.criterion(pred_last.reshape(-1, self.codebook_size + 1), code_last.reshape(-1)) / self.grad_acc_steps
        loss = loss_pred_last

        # update weight if accumulation of gradient is done
        update_grad = batch_idx % self.grad_acc_steps == self.grad_acc_steps - 1
        if update_grad:
            self.log("train/ce_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss
    
    def reco(self, x_0=None, x_last=None, code=None, masked_code=None, unmasked_code=None, mask=None):
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

        l_visual = [x_0[:,i] for i in range(x_0.size(1))]
        if code is not None:
            code = code.view(code.size(0), self.num_tokens, self.num_tokens)#.long()
            # Decoding reel code
            _x = self.ae.decode_code(torch.clamp(code, 0, self.codebook_size-1))
            if isinstance(_x, tuple):
                _x = _x[0]
            if mask is not None:
                # Decoding reel code with mask to hide
                mask = mask.view(code.size(0), 1, self.num_tokens, self.num_tokens).float()
                __x2 = _x * (1 - F.interpolate(mask, (self.image_size, self.image_size)).to(self.device))
                l_visual.append(__x2)
        if masked_code is not None:
            # Decoding masked code
            masked_code = masked_code.view(code.size(0), self.num_tokens, self.num_tokens)
            __x = self.ae.decode_code(torch.clamp(masked_code, 0, self.codebook_size-1))
            if isinstance(__x, tuple):
                __x = __x[0]
            l_visual.append(__x)

        if unmasked_code is not None:
            # Decoding predicted code
            unmasked_code = unmasked_code.view(code.size(0), self.num_tokens, self.num_tokens)
            ___x = self.ae.decode_code(torch.clamp(unmasked_code, 0, self.codebook_size-1))
            if isinstance(___x, tuple):
                ___x = ___x[0]
            l_visual.append(___x)

        return (torch.cat(l_visual, dim=0) + 1.0) / 2.0

    @torch.no_grad()
    def sample(self, x_0=None, code_0=None, init_code=None, sm_temp=1, w=0, randomize="linear", r_temp=4.5, schedule_mode=None, num_steps=None):
        """ Generate sample with the MaskGIT model
           :param
            x_0 / code_0   -> torch.FloatTensor: condition frames either in image or code format
            init_code       -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            sm_temp         -> float:            the temperature before softmax
            randomize       -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp          -> float:            temperature for the randomness
            schedule_mode   -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            num_steps:      -> int:              number of step for the decoding
           :return ... TODO
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks

        assert x_0 is not None or code_0 is not None, 'Either x_0 or code_0 must be provided'
        assert x_0 is None or code_0 is None, 'Only one of x_0 or code_0 must be provided'

        if x_0 is not None:
            _, code_0 = self.encode_frames(x_0)
        num_samples, num_cond_frames = code_0.shape[:2]
        code_0 = code_0.view(num_samples, num_cond_frames, self.num_tokens, self.num_tokens)

        if init_code is not None:  # Start with a pre-define code
            code = init_code
            mask = (init_code == self.codebook_size).float().view(num_samples, self.num_tokens*self.num_tokens)
        else:  # Initialize a code
            if self.mask_value < 0:  # Code initialize with random tokens
                code_last = torch.randint(0, self.codebook_size, (num_samples, self.num_tokens, self.num_tokens)).to(self.device)
            else:  # Code initialize with masked tokens
                code_last = torch.full((num_samples, self.num_tokens, self.num_tokens), self.mask_value).to(self.device)
            # start with a full mask           
            mask = torch.ones(num_samples, self.num_tokens*self.num_tokens).to(self.device)

        code_combined = torch.cat([code_0, code_last.unsqueeze(1)], dim=1)

        # Instantiate scheduler
        scheduler = self.scheduler.adap_sche(num_steps, schedule_mode)

        # Beginning of sampling, t = number of token to predict a step "indice"
        logits_last = torch.full((num_samples, self.num_tokens, self.num_tokens, self.codebook_size+1), -math.inf).to(self.device)
        for indice, t in enumerate(scheduler):
            if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                t = int(mask.sum().item())

            if mask.sum() == 0:  # Break if code is fully predicted
                break

            if w != 0:
                raise NotImplementedError('Not clear how/whether to do guidance for next frame')
            else:
                logit = self.vit(code_combined, drop_frame=None, drop_token=None)
            # if logit is all prediction
            logit = logit[:, -self.num_tokens*self.num_tokens:]

            prob = torch.softmax(logit * sm_temp, -1)
            # Sample the code from the softmax prediction
            distri = torch.distributions.Categorical(probs=prob)
            pred_code = distri.sample()
            
            conf = torch.gather(prob, 2, pred_code.view(num_samples, self.num_tokens*self.num_tokens, 1))

            if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                ratio = (indice / (len(scheduler)-1))
                rand = r_temp * np.random.gumbel(size=(num_samples, self.num_tokens*self.num_tokens)) * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device)
            elif randomize == "constant":  # add gumbel noise decreasing over the sampling process
                #ratio = (indice / (len(scheduler)-1))
                rand = r_temp * np.random.gumbel(size=(num_samples, self.num_tokens*self.num_tokens))# * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device)
            elif randomize == "warm_up":  # chose random sample for the 2 first steps
                conf = torch.rand_like(conf.squeeze()) if indice < 4 else conf.squeeze()
            elif randomize == "random":   # chose random prediction at each step
                conf = torch.rand_like(conf.squeeze())

            # do not predict on already predicted tokens
            conf[~mask.bool()] = -math.inf

            # chose the predicted token with the highest confidence
            tresh_conf, indice_mask = torch.topk(conf.view(num_samples, -1), k=t, dim=-1)
            tresh_conf = tresh_conf[:, -1]

            # replace the chosen tokens
            conf = (conf >= tresh_conf.unsqueeze(-1)).view(num_samples, self.num_tokens, self.num_tokens)
            f_mask = (mask.view(num_samples, self.num_tokens, self.num_tokens).float() * conf.view(num_samples, self.num_tokens, self.num_tokens).float()).bool()
            code_last[f_mask] = pred_code.view(num_samples, self.num_tokens, self.num_tokens)[f_mask]
            code_combined = torch.cat([code_0, code_last.unsqueeze(1)], dim=1)
            logits_last[f_mask] = logit.view(num_samples, self.num_tokens, self.num_tokens, self.codebook_size+1)[f_mask]

            # update the mask
            for i_mask, ind_mask in enumerate(indice_mask):
                mask[i_mask, ind_mask] = 0
            l_codes.append(pred_code.view(num_samples, self.num_tokens, self.num_tokens).clone())
            l_mask.append(mask.view(num_samples, self.num_tokens, self.num_tokens).clone())

        # decode the final prediction
        code = torch.clamp(code_last, 0, self.codebook_size-1)
        x = self.ae.decode_code(code)
        x = ((x[0] if isinstance(x, tuple) else x) + 1.0) / 2.0
        self.vit.train()

        return {
            "sampled_image": x,
            "sampled_code": code,
            "logits": logits_last,
            "intermediate_codes": l_codes,
            "intermediate_masks": l_mask,
        }

    @torch.no_grad()
    def log_images(self, batch, **sample_kwargs):
        x = self.get_input(batch, 'image')
        N = min(4, x.size(0))
        
        # Generate sample for visualization
        x_0 = x[:, :-1]
        gen_sample = self.sample(x_0=x_0[:N])["sampled_image"]
        if isinstance(gen_sample, tuple): # if dino loss is included
            gen_sample = gen_sample[0]

        gen_sample_grid = vutils.make_grid(gen_sample, nrow=2, padding=2, normalize=False)

        # VQGAN encoding to img tokens
        emb, code = self.encode_frames(x)
        code_0, code_last = rearrange(code[:, :-1], "b f h w -> (b f) h w"), code[:, -1]
        ae_recon = self.ae.decode_code(torch.clamp(code_last, 0, self.codebook_size-1))
        ae_recon = ae_recon[0] if isinstance(ae_recon, tuple) else ae_recon
        ae_recon = (ae_recon + 1.0) / 2.0

        # Mask the last frame
        masked_code_last, mask_last = self.scheduler.get_mask_code(code_last, value=self.mask_value)

        # drop complete temporal -- Randomly mask out the code_0
        drop_frame = torch.empty(code_0.size(0)).uniform_() < self.drop_frame
        code_0[drop_frame] = self.codebook_size
        # drop random token -- Randomly mask out the previous frames codes
        drop_token = torch.empty(code_0.size(0), self.num_tokens, self.num_tokens).uniform_() < self.drop_token
        code_0[drop_token] = self.codebook_size

        # Concatenate the masked code and the drop label
        combined_code = torch.cat([code_0.view(x.size(0), -1, self.num_tokens, self.num_tokens), 
                                   masked_code_last.view(x.size(0), 1, self.num_tokens, self.num_tokens)], dim=1)
        
        pred = self.vit(combined_code, drop_frame=drop_frame, drop_token=drop_token)  # The unmasked tokens prediction
        pred_last = pred[:, -self.num_tokens*self.num_tokens:]

        # Show reconstruction
        unmasked_code = torch.softmax(pred_last, -1).max(-1)[1]
        reco_sample = self.reco(x_0=x_0[:N], code=code_last[:N], unmasked_code=unmasked_code[:N], mask=mask_last[:N])
        # reco_sample = vutils.make_grid(reco_sample.data, nrow=N, padding=2, normalize=False)
        
        chunks = torch.chunk(reco_sample, self.num_frames+1, dim=0)
        assert len(chunks) == self.num_frames + 1
        reco_sample = torch.cat([*chunks[:self.num_frames], ae_recon[:N], chunks[self.num_frames]], 0)
        reco_sample = vutils.make_grid(reco_sample.data, nrow=N, padding=2, normalize=False)

        ret = {"sampled": gen_sample_grid, "reconstructed": reco_sample, "sampled_raw": gen_sample}
        return ret
