import logging
import os
from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import CLIPTokenizer

import comfy.model_management
import comfy.sample
import comfy.utils


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        # self attention with mask
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
            need_weights=False,
            is_causal=True,
        )
        x = residual + attn_output

        # cross attention
        residual = x
        x = self.norm2(x)
        attn_output, _ = self.cross_attn(
            query=x, key=memory, value=memory, need_weights=False
        )
        x = residual + attn_output

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.mlp(x)

        return x


class Captioner(nn.Module):
    def __init__(self, config, vision_embed_dim: int, vocab_size: int):
        super().__init__()

        self.embed_dim = vision_embed_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = vocab_size
        self.max_length = config.max_length

        # projection from ViT dimension to decoder dimension
        self.projection = nn.Linear(self.embed_dim, self.hidden_dim)
        self.memory_pos_embedding = nn.Parameter(torch.zeros(1, 257, self.hidden_dim))

        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderBlock(config.hidden_dim, config.num_heads)
                for _ in range(config.num_blocks)
            ]
        )

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_length)
        self.register_buffer("causal_mask", causal_mask, persistent=False)


class CLIPtionModel(nn.Module):
    def __init__(self, config, clip, clip_vision):
        super().__init__()

        if not hasattr(clip, "cond_stage_model"):
            raise ValueError("CLIP is missing from model checkpoint")
        if not hasattr(clip.cond_stage_model, "clip_l"):
            raise ValueError("Must use model which includes CLIP-L")

        # store CLIP model references
        self.clip_text = clip
        self.clip_vision = clip_vision
        self.tokenizer = clip.tokenizer.clip_l.tokenizer
        self.text_model = clip.cond_stage_model.clip_l.transformer.text_model
        self.vision_model = clip_vision.model.vision_model

        # clip.cond_stage_model.clip_l.transformer.text_projection is empty
        # so load a copy from the CLIPtion safetensors file instead
        self.text_projection = nn.Linear(768, 768, bias=False)

        # create caption decoder
        self.captioner = Captioner(config, 1024, self.tokenizer.vocab_size)

        # use CLIP's token embeddings for output projection
        clip_embed_weight = self.text_model.embeddings.token_embedding.weight
        self.output_projection = nn.Linear(
            self.captioner.hidden_dim, self.tokenizer.vocab_size, bias=False
        )
        self.output_projection.weight = nn.Parameter(
            clip_embed_weight.clone(), requires_grad=False
        )

    def generate(
        self,
        images: torch.Tensor,
        seed: int = 42,
        temperature: float = 0.7,
        best_of: int = 1,
        ramble: bool = False,
    ) -> List[str]:
        device = comfy.model_management.get_torch_device()

        # encode images
        image_outputs = self.clip_vision.encode_image(images)
        image_features = image_outputs.last_hidden_state.to(device, dtype=torch.float16)
        image_embeds = image_outputs.image_embeds.to(device, dtype=torch.float16)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

        if image_features.size(2) != 1024:
            raise ValueError(
                f"Expected image features to have 1024 dimensions but got {image_features.size(2)}. Please ensure you are using CLIP L."
            )

        captions = []
        for image_idx in range(image_features.size(0)):
            features = image_features[image_idx : image_idx + 1]
            image_embed = image_embeds[image_idx : image_idx + 1]

            # generate candidates in parallel using single copy of features
            tokens = self._batch_generate(
                features,
                temperature,
                best_of,
                seed + image_idx,
                ramble=ramble,
            )

            if best_of == 1:
                text = self.tokenizer.decode(
                    tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                captions.append(text)
                continue

            # calculate CLIP similarity for each candidate
            candidates = []
            for token_seq in tokens:
                text = self.tokenizer.decode(
                    token_seq,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                text_embeds = self._text_to_embed(text, device)
                clip_sim = torch.sum(image_embed * text_embeds, dim=-1)[0]
                candidates.append((clip_sim.item(), text))

            # pick highest scoring candidate
            candidates.sort(key=lambda x: x[0], reverse=False)
            for score, text in candidates:
                logging.debug(f"({score:.3f}) {text}")
            captions.append(candidates[-1][1])

        return captions

    def generate_beam(
        self,
        images: torch.Tensor,
        beam_width: int = 4,
        ramble: bool = False,
    ) -> List[str]:
        device = comfy.model_management.get_torch_device()

        # get image features and embeddings
        image_outputs = self.clip_vision.encode_image(images)
        image_features = image_outputs.last_hidden_state.to(device, dtype=torch.float16)
        image_embeds = image_outputs.image_embeds.to(device, dtype=torch.float16)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

        if image_features.size(2) != 1024:
            raise ValueError(
                f"Expected image features to have 1024 dimensions but got {image_features.size(2)}. Please ensure you are using CLIP L."
            )

        captions = []
        for image_idx in range(image_features.size(0)):
            features = image_features[image_idx].unsqueeze(0)
            candidates = self._beam_search(
                features,
                image_embeds[image_idx : image_idx + 1],
                device,
                beam_width=beam_width,
                ramble=ramble,
            )
            # pick highest scoring candidate
            candidates.sort(key=lambda x: x[0], reverse=False)
            for score, text in candidates:
                logging.debug(f"({score:.3f}) {text}")
            captions.append(candidates[-1][1])
        return captions

    def get_tokenizer(self) -> CLIPTokenizer:
        return self.tokenizer

    def _batch_generate(
        self,
        image_features: torch.Tensor,
        temperature: float,
        batch_size: int,
        seed: Optional[int] = None,
        ramble: bool = False,
    ) -> torch.Tensor:
        tokenizer = self.tokenizer
        output_projection = self.output_projection
        token_embedding_ = self.text_model.embeddings.token_embedding
        pos_embedding_ = self.text_model.embeddings.position_embedding

        # project and add positional embeddings to image features
        memory = self.captioner.projection(image_features)
        memory = memory + self.captioner.memory_pos_embedding
        memory = memory.repeat(batch_size, 1, 1)

        # initialize sequences with EOS tokens and BOS at start
        sequences = torch.full(
            (batch_size, self.captioner.max_length),
            tokenizer.eos_token_id,
            dtype=torch.long,
            device=image_features.device,
        )
        sequences[:, 0] = tokenizer.bos_token_id
        current_length = 1

        # set up random generator
        generator = torch.Generator(device=image_features.device)
        if seed is not None:
            generator.manual_seed(seed)

        # generate tokens until hitting max length or all sequences have EOS
        for current_length in range(1, self.captioner.max_length - 1):
            # embed current sequences
            token_embeddings = token_embedding_(sequences[:, :current_length])
            positions = torch.arange(current_length, device=sequences.device)
            pos_embeddings = pos_embedding_(positions)
            x = token_embeddings + pos_embeddings

            # pass through decoder layers
            mask = self.captioner.causal_mask[:current_length, :current_length]
            for layer in self.captioner.layers:
                x = layer(x, memory, self_attn_mask=mask)

            # get next token probabilities
            logits = output_projection(x[:, -1:])
            logits = logits / temperature

            # force EOS for sequences that hit EOS, prevent EOS for rambling sequences
            prev_is_eos = sequences[:, current_length - 1] == tokenizer.eos_token_id
            vocab_mask = torch.zeros_like(logits)
            vocab_mask[prev_is_eos, :, :] = float("-inf")
            vocab_mask[prev_is_eos, :, tokenizer.eos_token_id] = 0
            if ramble:
                vocab_mask[~prev_is_eos, :, tokenizer.eos_token_id] = float("-inf")
            logits = logits + vocab_mask

            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs.squeeze(1), 1, generator=generator)

            # add tokens to sequences
            sequences[:, current_length] = next_tokens.squeeze(-1)

            # stop if all sequences generated an EOS token
            if not ramble and (next_tokens == tokenizer.eos_token_id).all():
                break

        return sequences

    def _beam_search(
        self,
        image_features: torch.Tensor,
        image_embed: torch.Tensor,
        device: torch.device,
        beam_width: int = 5,
        ramble: bool = False,
    ):
        tokenizer = self.tokenizer
        captioner = self.captioner
        token_embedding = self.text_model.embeddings.token_embedding
        pos_embedding = self.text_model.embeddings.position_embedding
        vocab_size = tokenizer.vocab_size

        # project image features
        memory = captioner.projection(image_features)
        memory = memory + captioner.memory_pos_embedding

        # start with beam_width copies of BOS token
        current_tokens = torch.full(
            (beam_width, 1), tokenizer.bos_token_id, dtype=torch.long, device=device
        )

        # track sequence scores
        scores = torch.zeros(beam_width, device=device)

        for step in range(captioner.max_length - 2):
            # embed current tokens
            token_embeddings = token_embedding(current_tokens)
            positions = torch.arange(current_tokens.size(1), device=device)
            pos_embeddings = pos_embedding(positions)
            x = token_embeddings + pos_embeddings

            # run decoder layers
            seq_len = x.size(1)
            mask = captioner.causal_mask[:seq_len, :seq_len]
            for layer in captioner.layers:
                x = layer(x, memory.repeat(beam_width, 1, 1), self_attn_mask=mask)

            # get next token logits and log probabilities
            logits = self.output_projection(x[:, -1:])
            if ramble:
                logits[:, :, tokenizer.eos_token_id] = -float("inf")
            log_probs = F.log_softmax(logits, dim=-1)

            if step == 0:
                # pick top-k tokens for first step
                scores = log_probs.squeeze(1)[0]
                scores, indices = scores.topk(beam_width)
                current_tokens = torch.cat(
                    [current_tokens[0:1].repeat(beam_width, 1), indices.unsqueeze(1)],
                    dim=1,
                )
            else:
                # calculate scores for next tokens [beam_width x vocab_size]
                next_scores = scores.unsqueeze(1) + log_probs.squeeze(1)

                # force sequences to continue EOS after first one
                prev_is_eos = current_tokens[:, -1] == tokenizer.eos_token_id
                vocab_mask = torch.zeros_like(next_scores)
                vocab_mask[prev_is_eos] = float("-inf")
                vocab_mask[prev_is_eos, tokenizer.eos_token_id] = 0
                next_scores = next_scores + vocab_mask

                # pick top beam_width sequences
                next_scores = next_scores.view(-1)
                scores, indices = next_scores.topk(beam_width)
                beam_indices = indices // vocab_size  # which sequence each came from
                token_indices = indices % vocab_size  # which token to append

                current_tokens = torch.cat(
                    [current_tokens[beam_indices], token_indices.unsqueeze(1)], dim=1
                )

            # check if all beams ended with EOS
            if (current_tokens[:, -1] == tokenizer.eos_token_id).all():
                break

        # add final EOS token
        current_tokens = torch.cat(
            [current_tokens, torch.full((beam_width, 1), tokenizer.eos_token_id, device=device)],
            dim=1,
        )

        # rank final candidates by CLIP similarity
        candidates = []
        for idx in range(beam_width):
            text = tokenizer.decode(
                current_tokens[idx],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            text_embeds = self._text_to_embed(text, device)
            clip_sim = torch.sum(image_embed * text_embeds, dim=-1)[0]
            candidates.append((clip_sim.item(), text))
        return candidates

    def _text_to_embed(self, text: str, device: torch.device) -> torch.Tensor:
        tokens = self.clip_text.tokenize(text)
        self.clip_text.load_model()
        self.clip_text.cond_stage_model.reset_clip_options()
        self.clip_text.cond_stage_model.set_clip_options({"projected_pooled": False})
        clip_l = self.clip_text.cond_stage_model.clip_l
        _, pooled = clip_l.encode_token_weights(tokens["l"])
        text_embeds = self.text_projection(pooled.to(device, dtype=torch.float16))
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds


class CLIPtionLoader:
    CATEGORY = "pharmapsychotic"
    FUNCTION = "load"
    RETURN_TYPES = ("CLIPTION",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model used for encoding the text."},
                ),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    def load(self, clip, clip_vision):
        state_dict = {}
        file = "CLIPtion_20241219_fp16.safetensors"
        base_path = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(base_path, file)):
            model_path = os.path.join(base_path, file)
        else:
            repo_id = "pharmapsychotic/CLIPtion"
            revision = "15ee8cb77a902616478a033332011ff640e72277"
            model_path = hf_hub_download(
                repo_id=repo_id, filename=file, revision=revision
            )
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        tp_dict = {"weight": state_dict.pop("text_projection.weight")}

        config = SimpleNamespace(
            **{"hidden_dim": 768, "num_heads": 8, "num_blocks": 6, "max_length": 77}
        )
        model = CLIPtionModel(config, clip, clip_vision)
        model.captioner.load_state_dict(state_dict)
        model.text_projection.load_state_dict(tp_dict)
        model.eval()
        model.to(comfy.model_management.get_torch_device(), dtype=torch.float16)
        return (model,)


class CLIPtionGenerate:
    CATEGORY = "pharmapsychotic"
    FUNCTION = "caption"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CLIPTION", {"tooltip": "The CLIPtion model."}),
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The random seed used for creating the caption.",
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "tooltip": "Temperature for sampling."},
                ),
                "best_of": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of options to evaluate.",
                    },
                ),
                "ramble": ("BOOLEAN", {"default": False}),
            },
        }

    def caption(
        self,
        model: CLIPtionModel,
        image: torch.Tensor,
        seed: int,
        temperature: float = 0.7,
        best_of: int = 1,
        ramble: bool = False,
    ):
        with torch.inference_mode():
            captions = model.generate(image, seed, temperature, best_of, ramble)
        return (captions,)


class CLIPtionBeamSearch:
    CATEGORY = "pharmapsychotic"
    FUNCTION = "caption"
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CLIPTION", {"tooltip": "The CLIPtion model."}),
                "image": ("IMAGE",),
                "beam_width": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of beams to maintain during search.",
                    },
                ),
            },
            "optional": {
                "ramble": ("BOOLEAN", {"default": False}),
            },
        }

    def caption(
        self,
        model: CLIPtionModel,
        image: torch.Tensor,
        beam_width: int = 4,
        ramble: bool = False,
    ):
        with torch.inference_mode():
            captions = model.generate_beam(image, beam_width, ramble)
        return (captions,)


NODE_CLASS_MAPPINGS = {
    "CLIPtionBeamSearch": CLIPtionBeamSearch,
    "CLIPtionGenerate": CLIPtionGenerate,
    "CLIPtionLoader": CLIPtionLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPtionBeamSearch": "CLIPtion Beam Search",
    "CLIPtionGenerate": "CLIPtion Generate",
    "CLIPtionLoader": "CLIPtion Loader",
}
