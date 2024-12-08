import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from safetensors import safe_open
from transformers import CLIPTokenizer

import comfy.model_management
import comfy.sample
import comfy.utils
import folder_paths


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
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
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.gamma1 = nn.Parameter(torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(torch.ones(embed_dim))
        self.gamma3 = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x, memory, self_attn_mask=None):
        # self attention with mask
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=self_attn_mask,
            need_weights=False,
            is_causal=True,
        )
        x = residual + self.gamma1 * attn_output

        # cross attention
        residual = x
        x = self.norm2(x)
        attn_output, _ = self.cross_attn(
            query=x, key=memory, value=memory, need_weights=False
        )
        x = residual + self.gamma2 * attn_output

        # FFN
        residual = x
        x = self.norm3(x)
        x = residual + self.gamma3 * self.mlp(x)

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
        self.layers = nn.ModuleList([
            DecoderBlock(config.hidden_dim, config.num_heads)
            for i in range(config.num_blocks)
        ])

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_length)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, image_features, token_embeddings, output_projection):
        # project and add positional embeddings to image features
        memory = self.projection(image_features)
        memory = memory + self.memory_pos_embedding

        # get causal mask for self-attention
        seq_len = token_embeddings.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        # pass through decoder layers
        x = token_embeddings
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask=mask)

        # project to vocabulary
        logits = output_projection(x)
        return logits

    def generate(
        self,
        image_features,
        temperature,
        tokenizer,
        output_projection,
        token_embedding_,
        pos_embedding_,
    ):
        # project and add positional embeddings to image features
        memory = self.projection(image_features)
        memory = memory + self.memory_pos_embedding

        # start with BOS token
        batch_size = image_features.size(0)
        current_tokens = torch.full(
            (batch_size, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=image_features.device,
        )
        generated_tokens = [current_tokens]

        for _ in range(self.max_length - 2):
            # embed current tokens
            token_embeddings = token_embedding_(current_tokens)
            positions = torch.arange(
                current_tokens.size(1), device=current_tokens.device
            )
            pos_embeddings = pos_embedding_(positions)
            x = token_embeddings + pos_embeddings

            # get causal mask for self-attention
            seq_len = x.size(1)
            mask = self.causal_mask[:seq_len, :seq_len]

            # pass through decoder layers
            for layer in self.layers:
                x = layer(x, memory, self_attn_mask=mask)

            # get next token
            logits = output_projection(x[:, -1:])
            logits = logits / temperature  # temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1)

            generated_tokens.append(next_token)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

            # stop if all sequences have EOS token
            if (next_token == tokenizer.eos_token_id).all():
                break

        return torch.cat(generated_tokens, dim=1)


class CLIPtionModel(nn.Module):
    def __init__(self, config, clip, clip_vision):
        super().__init__()

        # store CLIP models
        self.tokenizer = clip.tokenizer.clip_l.tokenizer
        self.text_model = clip.cond_stage_model.clip_l.transformer.text_model
        self.vision_model = clip_vision.model.vision_model

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

    def forward(self, images, captions=None):
        vision_outputs = self.vision_model(images)
        image_features = vision_outputs.last_hidden_state

        token_embeddings = None
        if captions is not None:
            token_embeddings = self.text_model.embeddings.token_embedding(captions)
            positions = torch.arange(captions.size(1), device=captions.device)
            pos_embeddings = self.text_model.embeddings.position_embedding(positions)
            token_embeddings = token_embeddings + pos_embeddings

        return self.captioner(image_features, token_embeddings, self.output_projection)

    def get_tokenizer(self) -> CLIPTokenizer:
        return self.tokenizer

    def generate(self, images, temperature=0.7):
        vision_outputs = self.vision_model(images)
        image_features = vision_outputs.last_hidden_state
        return self.captioner.generate(
            image_features,
            temperature,
            self.tokenizer,
            self.output_projection,
            self.text_model.embeddings.token_embedding,
            self.text_model.embeddings.position_embedding,
        )


class CLIPtionLoader:
    CATEGORY = "pharmapsychotic"
    FUNCTION = "load"
    RETURN_TYPES = ("CLIPTION_MODEL",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "clip_vision": ("CLIP_VISION",),
            }
        }
    
    def load(self, clip, clip_vision):
        state_dict = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "CLIPtion_ViTL_20241129_fp16.safetensors")
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        config = SimpleNamespace(**{
            'hidden_dim': 768,
            'num_heads': 8,
            'num_blocks': 6,
            'max_length': 77
        })
        model = CLIPtionModel(config, clip, clip_vision)
        model.captioner.load_state_dict(state_dict)
        model.eval()
        model.to(comfy.model_management.get_torch_device())

        return (model,)


class CLIPtion:
    CATEGORY = "pharmapsychotic"
    FUNCTION = "caption"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CLIPTION_MODEL", {"tooltip": "The CLIPtion model."}),
                "image": ("IMAGE", ),
            }
        }

    def caption(self, model, image):
        return ("this is the caption",)


NODE_CLASS_MAPPINGS = {
    "CLIPtionLoader": CLIPtionLoader,
    "CLIPtion": CLIPtion,
}
