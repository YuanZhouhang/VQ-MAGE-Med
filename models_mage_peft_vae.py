from functools import partial

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed

from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
import torch.distributed as dist


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Adapter(nn.Module):
    def __init__(self, d_model=None, bottleneck=None, dropout=0.0, init_option='bert', adapter_scalar='1.0', adapter_layernorm_option='in'):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == 'in' or adapter_layernorm_option == 'out':
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.droupout = dropout
        if init_option == 'bert':
            raise NotImplementedError
        elif init_option == 'lora':
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.droupout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.adapt = Adapter(d_model=dim, bottleneck=mlp_hidden_dim, dropout=0.0, init_option='lora', adapter_scalar='0.1', adapter_layernorm_option='none')

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            adapt_x = self.adapt(x)
            # if not torch.isnan(x).any():
            #     print('x', torch.mean(torch.abs(x)))
            # if not torch.isnan(adapt_x).any():
            #     print('adapt_x', torch.mean(torch.abs(adapt_x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x = x + adapt_x
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
        self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits


class QuantizerEMA(nn.Module):
    def __init__(self, embedding_dim, num_embeddings: int = 256, commitment_loss_factor: float = 0.25, decay: float = 0.99, epsilon=1e-5):
        super().__init__()

        # commitment_loss_factor: float = 0.25
        # quantization_loss_factor: float = 1.0
        # num_embeddings: int = 512
        # use_ema: bool = False
        # decay: float = 0.99

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_factor = commitment_loss_factor
        self.decay = decay

        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        embeddings.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer(
            "ema_embed", torch.zeros(self.num_embeddings, self.embedding_dim)
        )
        self.ema_embed.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.register_buffer("embeddings", embeddings)

    def forward(self, z: torch.Tensor, uses_ddp: bool = False):

        distances = (
            (z.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings ** 2).sum(dim=-1)
            - 2 * z.reshape(-1, self.embedding_dim) @ self.embeddings.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        quantized_indices = closest.reshape(z.shape[0], z.shape[1], z.shape[2])

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings
        quantized = quantized.reshape_as(z)

        if self.training:

            n_i = torch.sum(one_hot_encoding, dim=0)

            if uses_ddp:
                dist.all_reduce(n_i)

            self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

            dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)

            if uses_ddp:
                dist.all_reduce(dw)

            ema_embed = self.ema_embed * self.decay + dw * (1 - self.decay)

            n = torch.sum(self.cluster_size)

            self.cluster_size = (
                (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            )

            self.embeddings.data.copy_(ema_embed / self.cluster_size.unsqueeze(-1))
            self.ema_embed.data.copy_(ema_embed)

        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="mean",
        )

        quantized = z + (quantized - z).detach()

        loss = commitment_loss * self.commitment_loss_factor
        # quantized = quantized.permute(0, 3, 1, 2)

        return loss, quantized.squeeze(), quantized_indices.unsqueeze(1)


class Metric_MLP(nn.Module):
    """ Metric MLP
    """
    def __init__(self, in_dim, latent_dim):
        self.input_dim = in_dim
        self.latent_dim = latent_dim

        self.layers = nn.Sequential(nn.Linear(np.prod(self.input_dim), 400), nn.ReLU())
        self.diag = nn.Linear(400, self.latent_dim)
        k = int(self.latent_dim * (self.latent_dim + 1) / 2)
        self.lower = nn.Linear(400, k)

    def forward(self, x):
        h1 = self.layers(x.reshape(-1, np.prod(self.input_dim)))
        h21, h22 = self.diag(h1), self.lower(h1)

        L = torch.zeros(x.shape[0], self.latent_dim, self.latent_dim).to(x.device)
        indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)

        L[:, indices[0], indices[1]] = h22
        L = L+torch.diag_embed(h21.exp())
        return L



class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt', vae_set='vqvae'):
        super().__init__()

        # --------------------------------------------------------------------------
        # VQGAN specifics
        config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        self.codebook_size = config.params.n_embed
        vocab_size = self.codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        self.fake_class_label = self.codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=256+1,
                                        dropout=0.1)

        # MAGE variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        # --------------------------------------------------------------------------
        # MAGE encoder specifics
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAGE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # VAE
        self.vae_set = vae_set
        if vae_set == 'vqvae':
            self.quantizier = QuantizerEMA(num_embeddings=512, embedding_dim=768)
        # self.metric = Metric_MLP()


        self.initialize_weights()
        self._freeze_stages()

    def _freeze_stages(self):
        for param in self.token_emb.parameters():
            param.requires_grad = False
        self.pos_embed.requires_grad = False
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for name, parametr in self.blocks.named_parameters():
            if 'adapt' in name:
                parametr.requires_grad = True
            else:
                parametr.requires_grad = False
        # self.norm.eval()

        for param in self.decoder_embed.parameters():
            param.requires_grad = False
        self.mask_token.requires_grad = False
        self.decoder_pos_embed.requires_grad = False
        self.decoder_pos_embed_learned.requires_grad = False
        for name, parametr in self.decoder_blocks.named_parameters():
            if 'adapt' in name:
                parametr.requires_grad = True
            else:
                parametr.requires_grad = False
        # self.decoder_norm.eval()
        # self.decoder_pred.eval()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(x)

        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)
        gt_indices = token_indices.clone().detach().long()

        # masking
        bsz, seq_len = token_indices.size()
        mask_ratio_min = self.mask_ratio_min
        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=x.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break
            else:
                print("Rerandom the noise!")
        # print(mask_rate, num_dropped_tokens, num_masked_tokens, token_drop_mask.sum(dim=1), token_all_mask.sum(dim=1))
        token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        # print("Masekd num token:", torch.sum(token_indices == self.mask_token_label, dim=1))

        # concate class token
        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = self.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        token_indices = token_indices.long()
        # bert embedding
        input_embeddings = self.token_emb(token_indices)
        # print("Input embedding shape:", input_embeddings.shape)
        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)
        # print("Input embedding after drop shape:", input_embeddings_after_drop.shape)

        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # print("Encoder representation shape:", x.shape)

        return x, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)
        # print("Logits shape:", x.shape)

        return x

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits[:, 1:, :self.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        return loss

    def forward_vq_loss(self, latent):
        # embeddings = latent.reshape(latent.shape[0], -1)
        # embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
        # embeddings = embeddings.permute(0, 2, 3, 1)
        embeddings = latent[:, 0:1, :].unsqueeze(1)
        vq_loss, quantized_embed, _ = self.quantizier(embeddings)
        # print(quantized_embed.shape)
        return vq_loss, quantized_embed


    def forward_rh_loss(self, latent, gt_indices, logits, mask):
        if self.training:
            pass
        return 0

    def forward(self, imgs):
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(imgs)

        if self.vae_set == 'rhvae':
            # TODO: rhvae can be used in theory, but the preliminary experimental results are not good and need to be verified.
            pass
            # rh_loss = self.forward_rh_loss(latent, gt_indices, logits, token_all_mask)
            # loss = original_loss + rh_loss
        elif self.vae_set == 'vqvae':
            vq_loss, quantized_embed = self.forward_vq_loss(latent)
            latent_after_vq = torch.concat([quantized_embed.unsqueeze(1), latent[:, 1:, :]], dim=1)
            logits = self.forward_decoder(latent_after_vq, token_drop_mask, token_all_mask)
            original_loss = self.forward_loss(gt_indices, logits, token_all_mask)
            loss = original_loss + vq_loss
            # print("VQ loss:", vq_loss, "Original loss:", original_loss)
        else:
            logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
            original_loss = self.forward_loss(gt_indices, logits, token_all_mask)
        return loss, imgs, token_all_mask


def mage_vit_base_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mage_vit_large_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
