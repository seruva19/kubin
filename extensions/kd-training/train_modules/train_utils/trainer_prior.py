import sys

import copy
import functools
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from train_modules.train_utils.utils import generate_mask, get_image_mask
import clip


def encode_text(tok, clip_model):
    with torch.no_grad():
        x = clip_model.token_embedding(tok).type(clip_model.dtype)
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        txt_feat_seq = x
        txt_feat = (
            x[torch.arange(x.shape[0]), tok.argmax(dim=-1)] @ clip_model.text_projection
        )
        txt_feat, txt_feat_seq = txt_feat.float(), txt_feat_seq.float()
        return txt_feat, txt_feat_seq


def encode_image(image, clip_model, clip_mean, clip_std):
    with torch.no_grad():
        return (clip_model.encode_image(image).float() - clip_mean) / clip_std


def train_prior(
    model,
    diffusion,
    clip_model,
    optimizer,
    lr_scheduler=None,
    schedule_sampler=None,
    train_loader=None,
    val_loader=None,
    num_epochs=2,
    save_every=1000,
    save_epoch=1,
    save_name="model",
    save_path="",
    device="cuda:0",
):
    assert train_loader is not None
    assert schedule_sampler is not None
    train_epoch = 0

    for epoch in range(num_epochs):
        train_epoch += 1
        train_step = 0
        progress = tqdm(
            total=len(train_loader),
            desc=f"finetuning prior model, epoch {train_epoch}",
            position=0,
            leave=True,
        )
        for batch in train_loader:
            train_step += 1

            optimizer.zero_grad()
            image, cond = batch
            image = image.to(device)
            for key in cond.keys():
                cond[key] = cond[key].to(device)
            image = encode_image(
                image, clip_model, model.clip_mean.to(device), model.clip_std.to(device)
            )
            txt_feat, txt_feat_seq = encode_text(cond["tokens"], clip_model)
            cond = {
                "text_emb": txt_feat,
                "text_enc": txt_feat_seq,
                "mask": cond["mask"],
                "causal_mask": model.causal_mask,
            }
            t, weights = schedule_sampler.sample(image.shape[0], image.device)
            compute_losses = functools.partial(
                diffusion.training_losses,
                model.model,
                image,
                t,
                model_kwargs=cond,
            )
            losses = compute_losses()
            loss = losses["loss"].mean()  # type: ignore
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            train_step += 1
            if save_every != 0 and train_step % save_every == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_path,
                        save_name + f"{epoch + 1}_{str(train_step)}" + ".ckpt",
                    ),
                )
            progress.set_postfix({"step": train_step + 1, "loss": loss.item()})
            progress.update()

        if (train_epoch == num_epochs) or (
            save_epoch != 0 and train_epoch % save_epoch
        ) == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_path, save_name + f"{epoch + 1}" + ".ckpt"),
            )
