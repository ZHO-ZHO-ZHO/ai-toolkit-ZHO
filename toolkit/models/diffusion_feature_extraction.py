import torch
import os
from torch import nn
from safetensors.torch import load_file
import torch.nn.functional as F
from diffusers import AutoencoderTiny
from transformers import SiglipImageProcessor, SiglipVisionModel
import lpips

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels,
                              1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x + identity)
        return x


class DiffusionFeatureExtractor2(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.version = 2

        # Path 1: Upsample to 512x512 (1, 64, 512, 512)
        self.up_path = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(64, 64),
            nn.Conv2d(64, 64, 3, padding=1),
        ])

        # Path 2: Upsample to 256x256 (1, 128, 256, 256)
        self.path2 = nn.ModuleList([
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(128, 128),
            nn.Conv2d(128, 128, 3, padding=1),
        ])

        # Path 3: Upsample to 128x128 (1, 256, 128, 128)
        self.path3 = nn.ModuleList([
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(256, 256),
            nn.Conv2d(256, 256, 3, padding=1)
        ])

        # Path 4: Original size (1, 512, 64, 64)
        self.path4 = nn.ModuleList([
            nn.Conv2d(in_channels, 512, 3, padding=1),
            ResBlock(512, 512),
            ResBlock(512, 512),
            nn.Conv2d(512, 512, 3, padding=1)
        ])

        # Path 5: Downsample to 32x32 (1, 512, 32, 32)
        self.path5 = nn.ModuleList([
            nn.Conv2d(in_channels, 512, 3, padding=1),
            ResBlock(512, 512),
            nn.AvgPool2d(2),
            ResBlock(512, 512),
            nn.Conv2d(512, 512, 3, padding=1)
        ])

    def forward(self, x):
        outputs = []

        # Path 1: 512x512
        x1 = x
        for layer in self.up_path:
            x1 = layer(x1)
        outputs.append(x1)  # [1, 64, 512, 512]

        # Path 2: 256x256
        x2 = x
        for layer in self.path2:
            x2 = layer(x2)
        outputs.append(x2)  # [1, 128, 256, 256]

        # Path 3: 128x128
        x3 = x
        for layer in self.path3:
            x3 = layer(x3)
        outputs.append(x3)  # [1, 256, 128, 128]

        # Path 4: 64x64
        x4 = x
        for layer in self.path4:
            x4 = layer(x4)
        outputs.append(x4)  # [1, 512, 64, 64]

        # Path 5: 32x32
        x5 = x
        for layer in self.path5:
            x5 = layer(x5)
        outputs.append(x5)  # [1, 512, 32, 32]

        return outputs


class DFEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x + x_in
        return x


class DiffusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.version = 1
        num_blocks = 6
        self.conv_in = nn.Conv2d(in_channels, 512, 1)
        self.blocks = nn.ModuleList([DFEBlock(512) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_out(x)
        return x


class DiffusionFeatureExtractor3(nn.Module):
    def __init__(self, device=torch.device("cuda"), dtype=torch.bfloat16):
        super().__init__()
        self.version = 3
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taef1", torch_dtype=torch.bfloat16)
        self.vae = vae
        image_encoder_path = "google/siglip-so400m-patch14-384"
        try:
            self.image_processor = SiglipImageProcessor.from_pretrained(
                image_encoder_path)
        except EnvironmentError:
            self.image_processor = SiglipImageProcessor()
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            image_encoder_path,
            ignore_mismatched_sizes=True
        ).to(device, dtype=dtype)

        self.lpips_model = lpips_model = lpips.LPIPS(net='vgg')
        self.lpips_model = lpips_model.to(device, dtype=torch.float32)
        self.losses = {}
        self.log_every = 100
        self.step = 0
        
    def get_siglip_features(self, tensors_0_1):
        dtype = torch.bfloat16
        device = self.vae.device
        # resize to 384x384
        images = F.interpolate(tensors_0_1, size=(384, 384),
                               mode='bicubic', align_corners=False)

        mean = torch.tensor(self.image_processor.image_mean).to(
            device, dtype=dtype
        ).detach()
        std = torch.tensor(self.image_processor.image_std).to(
            device, dtype=dtype
        ).detach()
        # tensors_0_1 = torch.clip((255. * tensors_0_1), 0, 255).round() / 255.0
        clip_image = (
            images - mean.view([1, 3, 1, 1])) / std.view([1, 3, 1, 1])
        id_embeds = self.vision_encoder(
            clip_image,
            output_hidden_states=True,
        )

        last_hidden_state = id_embeds['last_hidden_state']
        return last_hidden_state
    
    def get_lpips_features(self, tensors_0_1):
        device = self.vae.device
        tensors_n1p1 = (tensors_0_1 * 2) - 1
        def get_lpips_features(img):  # -1 to 1
            in0_input = self.lpips_model.scaling_layer(img)
            outs0 = self.lpips_model.net.forward(in0_input)

            feats0 = {}

            feats_list = []
            for kk in range(self.lpips_model.L):
                feats0[kk] = lpips.normalize_tensor(outs0[kk])
                feats_list.append(feats0[kk])

            # 512 in
            # vgg
            # 0 torch.Size([1, 64, 512, 512])
            # 1 torch.Size([1, 128, 256, 256])
            # 2 torch.Size([1, 256, 128, 128])
            # 3 torch.Size([1, 512, 64, 64])
            # 4 torch.Size([1, 512, 32, 32])

            return feats_list

        # do lpips
        lpips_feat_list = [x.detach() for x in get_lpips_features(
            tensors_n1p1.to(device, dtype=torch.float32))]
        
        return lpips_feat_list
        

    def forward(
        self, 
        noise_pred,
        noisy_latents,
        timesteps, 
        batch: DataLoaderBatchDTO, 
        scheduler: CustomFlowMatchEulerDiscreteScheduler,
        lpips_weight=20.0,
        clip_weight=0.1,
        pixel_weight=1.0
    ):
        dtype = torch.bfloat16
        device = self.vae.device
        
        # first we step the scheduler from current timestep to the very end for a full denoise
        bs = noise_pred.shape[0]
        noise_pred_chunks = torch.chunk(noise_pred, bs)
        timestep_chunks = torch.chunk(timesteps, bs)
        noisy_latent_chunks = torch.chunk(noisy_latents, bs)
        stepped_chunks = []
        for idx in range(bs):
            model_output = noise_pred_chunks[idx]
            timestep = timestep_chunks[idx]
            scheduler._step_index = None
            scheduler._init_step_index(timestep)
            sample = noisy_latent_chunks[idx].to(torch.float32)
            
            sigma = scheduler.sigmas[scheduler.step_index]
            sigma_next = scheduler.sigmas[-1] # use last sigma for final step
            prev_sample = sample + (sigma_next - sigma) * model_output
            stepped_chunks.append(prev_sample)
        
        stepped_latents = torch.cat(stepped_chunks, dim=0)
            
        latents = stepped_latents.to(self.vae.device, dtype=self.vae.dtype)
        
        latents = (
            latents / self.vae.config['scaling_factor']) + self.vae.config['shift_factor']
        tensors_n1p1 = self.vae.decode(latents).sample  # -1 to 1
        
        pred_images = (tensors_n1p1 + 1) / 2  # 0 to 1
        
        pred_clip_output = self.get_siglip_features(pred_images)
        lpips_feat_list_pred = self.get_lpips_features(pred_images.float())
        
        with torch.no_grad():
            target_img = batch.tensor.to(device, dtype=dtype)
            # go from -1 to 1 to 0 to 1
            target_img = (target_img + 1) / 2
            target_clip_output = self.get_siglip_features(target_img).detach()
            lpips_feat_list_target = self.get_lpips_features(target_img.float())
            
        clip_loss = torch.nn.functional.mse_loss(
            pred_clip_output.float(), target_clip_output.float()
        ) * clip_weight
        
        if 'clip_loss' not in self.losses:
            self.losses['clip_loss'] = clip_loss.item()
        else:
            self.losses['clip_loss'] += clip_loss.item()
        
        total_loss = clip_loss
        
        lpips_loss = 0
        for idx, lpips_feat in enumerate(lpips_feat_list_pred):
            lpips_loss += torch.nn.functional.mse_loss(
                lpips_feat.float(), lpips_feat_list_target[idx].float()
            ) * lpips_weight
            
        if 'lpips_loss' not in self.losses:
            self.losses['lpips_loss'] = lpips_loss.item()
        else:
            self.losses['lpips_loss'] += lpips_loss.item()
            
        total_loss += lpips_loss
            
        mse_loss = torch.nn.functional.mse_loss(
            stepped_latents.float(), batch.latents.float()
        ) * pixel_weight
        
        if 'pixel_loss' not in self.losses:
            self.losses['pixel_loss'] = mse_loss.item()
        else:
            self.losses['pixel_loss'] += mse_loss.item()
            
        if self.step % self.log_every == 0 and self.step > 0:
            print(f"DFE losses:")
            for key in self.losses:
                self.losses[key] /= self.log_every
                # print in 2.000e-01 format
                print(f" - {key}: {self.losses[key]:.3e}")
            self.losses[key] = 0.0
        
        total_loss += mse_loss
        self.step += 1
        
        return total_loss


def load_dfe(model_path) -> DiffusionFeatureExtractor:
    if model_path == "v3":
        dfe = DiffusionFeatureExtractor3()
        dfe.eval()
        return dfe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # if it ende with safetensors
    if model_path.endswith('.safetensors'):
        state_dict = load_file(model_path)
    else:
        state_dict = torch.load(model_path, weights_only=True)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

    if 'conv_in.weight' in state_dict:
        dfe = DiffusionFeatureExtractor()
    else:
        dfe = DiffusionFeatureExtractor2()

    dfe.load_state_dict(state_dict)
    dfe.eval()
    return dfe
