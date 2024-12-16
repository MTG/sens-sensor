import dataclasses
import logging
from pathlib import Path
import librosa
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

from CLAP.src.laion_clap.clap_module.htsat import create_htsat_model
from CLAP.src.laion_clap.clap_module.model import MLPLayers, CLAPAudioCfp
from CLAP.src.laion_clap.clap_module.pann_model import create_pann_model
from CLAP.src.laion_clap.training.data import (
    int16_to_float32,
    float32_to_int16,
    get_audio_features,
)
import CLAP.src.laion_clap.clap_module.factory as factory
from CLAP.src.laion_clap.clap_module import convert_weights_to_fp16


class AudioTower(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        audio_cfg: CLAPAudioCfp,
        enable_fusion: bool = False,
        fusion_type: str = "None",
        joint_embed_shape: int = 512,
        mlp_act: str = "relu",
    ):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLAPAudioCfp(**audio_cfg)

        self.audio_cfg = audio_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape
        self.mlp_act = mlp_act

        if mlp_act == "relu":
            mlp_act_layer = nn.ReLU()
        elif mlp_act == "gelu":
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # audio branch
        # audio branch parameters
        if audio_cfg.model_type == "PANN":
            self.audio_branch = create_pann_model(audio_cfg, enable_fusion, fusion_type)
        elif audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(
                audio_cfg, enable_fusion, fusion_type
            )
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")

        # audio branch parameters
        self.audio_transform = MLPLayers(
            units=[
                self.joint_embed_shape,
                self.joint_embed_shape,
                self.joint_embed_shape,
            ],
            dropout=0.1,
        )

        # =============================================================================
        self.audio_projection = nn.Sequential(
            nn.Linear(embed_dim, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
        )

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio, device):
        return self.audio_branch(
            audio, mixup_lambda=None, device=device
        )  # mix lambda needs to add

    def get_audio_embedding(self, data):
        """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: torch.Tensor
            a tensor of audio_embeds (N, D)

        """
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(
                device
            )
        audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
        audio_embeds = self.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    def get_audio_embedding_from_filelist(self, x, use_tensor=False):
        """get audio embeddings from the audio file list

        Parameters
        ----------
        x: List[str] (N,):
            an audio file list to extract features, audio files can have different lengths (as we have the feature fusion machanism)
        use_tensor: boolean:
            if True, it will return the torch tensor, preserving the gradient (default: False).
        Returns
        ----------
        audio_embed : numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """
        self.eval()
        audio_input = []
        for f in x:
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, _ = librosa.load(f, sr=48000)
            # quantize
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            print("waveform", audio_waveform.mean())
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict,
                audio_waveform,
                480000,
                data_truncating="fusion" if self.enable_fusion else "rand_trunc",
                data_filling="repeatpad",
                audio_cfg=dataclasses.asdict(self.audio_cfg),
                require_grad=audio_waveform.requires_grad,
            )
            audio_input.append(temp_dict)
        print("feat waveform", audio_input[0]["waveform"].mean())
        audio_embed = self.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed

    def get_audio_embedding_from_data(self, x, use_tensor=False):
        """get audio embeddings from the audio data

        Parameters
        ----------
        x: np.darray | torch.Tensor (N,T):
            audio data, must be mono audio tracks.
        use_tensor: boolean:
            if True, x should be the tensor input and the output will be the tesnor, preserving the gradient (default: False).
            Note that if 'use tensor' is set to True, it will not do the quantize of the audio waveform (otherwise the gradient will not be preserved).
        Returns
        ----------
        audio embed: numpy.darray | torch.Tensor (N,D):
            audio embeddings that extracted from audio files
        """
        self.eval()
        audio_input = []
        for audio_waveform in x:
            # quantize
            if not use_tensor:
                audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
                audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict,
                audio_waveform,
                480000,
                data_truncating="fusion" if self.enable_fusion else "rand_trunc",
                data_filling="repeatpad",
                audio_cfg=dataclasses.asdict(self.audio_cfg),
                require_grad=audio_waveform.requires_grad,
            )
            audio_input.append(temp_dict)
        audio_embed = self.get_audio_embedding(audio_input)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed


def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # removing position_ids to maintain compatibility with latest transformers update
        """if version.parse(transformers.__version__) >= version.parse("4.31.0"): 
            del state_dict["text_branch.embeddings.position_ids"]"""
    # for k in state_dict:
    #     if k.startswith('transformer'):
    #         v = state_dict.pop(k)
    #         state_dict['text_branch.' + k[12:]] = v
    return state_dict


def create_tower(
    ckpt: str,
    amodel="HTSAT-tiny",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    enable_fusion: bool = False,
    fusion_type: str = "None",
):
    amodel_name = amodel.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names

    if amodel_name in factory._MODEL_CONFIGS:
        logging.info(f"Loading {amodel_name} model config.")
        model_cfg = deepcopy(factory._MODEL_CONFIGS[amodel_name])
    else:
        logging.error(
            f"Model config for {amodel_name} not found; available models {factory.list_models()}."
        )
        raise RuntimeError(f"Model config for {amodel_name} not found.")

    if enable_fusion:
        fusion_type = "aff_2d"
    model_cfg["enable_fusion"] = enable_fusion
    model_cfg["fusion_type"] = fusion_type
    del model_cfg["text_cfg"]
    model = AudioTower(**model_cfg)

    model.to(device=device)
    if precision == "fp16":
        assert device.type != "cpu"
        convert_weights_to_fp16(model)

    if jit:
        model = torch.jit.script(model)

    state_dict = load_state_dict(ckpt, map_location=device)
    model.load_state_dict(state_dict)

    return model, model_cfg


def minify_checkpoint(ckpt: Path):
    # Check if the checkpoint file exists
    if not ckpt.exists():
        print(f"Error: Checkpoint file '{ckpt}' does not exist.")
        return
    print(f"Checkpoint '{ckpt}' exists. Proceeding with minification...")

    state_dict = factory.load_state_dict(ckpt, map_location="cpu")
    audio_state_dict = {
        k: v
        for k, v in state_dict.items()
        if not (k.startswith("text") or k.endswith("_t"))
    }

    new_path = ckpt.with_name(ckpt.stem + "_audio_only").with_suffix(ckpt.suffix)
    new_ckpt = {"state_dict": audio_state_dict}
    torch.save(new_ckpt, new_path)
