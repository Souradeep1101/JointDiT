# data_loader/jointdit_dataset.py
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class JointDiTDataset(Dataset):
    """
    Loads cached latents + optional CLIP conditionings.

    Returns a dict; only numeric tensor-like fields that exist in *all* items
    will survive your collate (strings like paths/captions are dropped there):

      img_firstframe_path : str (collate drops; useful for debug)
      v_latents           : (T, C, H, W)  video latents
      a_latents           : (1, C, H, W)  audio latents

      # CLIP image (back-compat + explicit name)
      clip_emb            : (1, D) or (D,)        [if present]
      clip_img_emb        : (1, D) or (D,)        [if present]

      # CLIP text (new; optional)
      clip_txt_tokens     : (1, 77) LongTensor    [if present]
      clip_txt_emb        : (1, D) or (D,)        [if present]

      captions            : List[str]             (collate drops)
      meta                : dict                  (collate drops)
    """

    def __init__(self, cache_root: str, split: str):
        self.cache_root = Path(cache_root)
        self.split = split
        meta_dir = self.cache_root / "meta" / split
        self.metas = sorted([p for p in meta_dir.glob("*.json")])
        if not self.metas:
            raise FileNotFoundError(f"No meta jsons under {meta_dir}")

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        mpath = self.metas[idx]
        meta = json.loads(Path(mpath).read_text())

        # --- basic paths ---
        img_path = meta.get("img_firstframe", "")
        v_lat_path = Path(meta["video_latents"])
        a_lat_path = Path(meta["audio_latents"])

        # --- load video/audio latents ---
        v = torch.load(v_lat_path, map_location="cpu", weights_only=False)
        v_lat = v["latents"]  # (T, C, H, W)

        a = torch.load(a_lat_path, map_location="cpu", weights_only=False)
        a_lat = a["latents"]  # (1, C, H, W)

        out = {
            "img_firstframe_path": img_path,
            "v_latents": v_lat,
            "a_latents": a_lat,
            "meta": meta,
        }

        # --- CLIP image embedding (back-compat: 'clip_file') ---
        clip_img_path = meta.get("clip_file", None)
        if clip_img_path:
            try:
                clip_img = torch.load(clip_img_path, map_location="cpu")
                clip_img_emb = clip_img.get("embedding", None)
                if isinstance(clip_img_emb, torch.Tensor):
                    out["clip_img_emb"] = clip_img_emb
                    out["clip_emb"] = clip_img_emb  # legacy alias
            except Exception:
                pass

        # --- CLIP text tokens (optional) ---
        txt_tok_path = meta.get("clip_text_tokens", None)
        if txt_tok_path:
            try:
                tok = torch.load(txt_tok_path, map_location="cpu")
                tokens = tok.get("tokens", None)
                if isinstance(tokens, torch.Tensor):
                    out["clip_txt_tokens"] = tokens
            except Exception:
                pass

        # --- CLIP text embedding (optional) ---
        txt_emb_path = meta.get("clip_text_emb", None)
        if txt_emb_path:
            try:
                t = torch.load(txt_emb_path, map_location="cpu")
                emb = t.get("embedding", None)
                if isinstance(emb, torch.Tensor):
                    out["clip_txt_emb"] = emb
            except Exception:
                pass

        # --- captions (optional, non-collatable) ---
        if "captions" in meta and isinstance(meta["captions"], list):
            out["captions"] = meta["captions"]

        return out
