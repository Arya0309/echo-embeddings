# echo_for_mteb.py
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser
from mteb.encoder_interface import PromptType
from tqdm.auto import tqdm
import sys, os
import torch
import torch.nn.functional as F
import numpy as np

# ---- 全域加速建議（Amp/TF32）----
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from contextlib import nullcontext

try:
    # 新寫法（PyTorch ≥ 2.0）
    from torch.amp import autocast as autocast_amp

    def autocast_cuda():
        return autocast_amp(device_type="cuda")

except Exception:
    # 舊版相容
    def autocast_cuda():
        import torch

        return torch.cuda.amp.autocast()


class EchoModel:
    def __init__(
        self,
        path_to_model,
        templates,
        max_length=300,
        pooling_strategy="mean",
        pad_to_multiple_of: int | None = 8,  # 有些 tokenizer 會吃這個參數
    ):
        self.templates = templates

        # decoder 模型作 encoder：保留你的 pipeline
        self.model = EchoEmbeddingsMistral.from_pretrained(path_to_model).eval()
        self.parser = EchoParser(
            path_to_model,
            templates,
            max_length=max_length,
        )
        self.pooling = EchoPooling(strategy=pooling_strategy)

        # 裝置與多卡
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count() if self.device.type == "cuda" else 0

        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
            if self.gpu_count > 1:
                # 讓 DataParallel 自動複製到各卡；輸入維持為單一 dict
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(self.gpu_count)), output_device=1
                )

        # 供 encode 用
        self.pad_to_multiple_of = pad_to_multiple_of

    def _template_key(self, prompt_type: PromptType | None) -> str:
        if prompt_type == PromptType.query and "query" in self.templates:
            return "query"
        if prompt_type == PromptType.passage and "document" in self.templates:
            return "document"
        if "document" in self.templates:
            return "document"
        if "query" in self.templates:
            return "query"
        return next(iter(self.templates.keys()))

    @torch.inference_mode()
    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:

        base_bs: int = int(kwargs.get("batch_size", 128))
        show_pbar: bool = bool(kwargs.get("show_progress_bar", True))

        # 更貼近作者：batch_size 乘上 GPU 數
        eff_bs = base_bs * max(1, self.gpu_count)

        tmpl = self._template_key(prompt_type)
        out_chunks = []

        indices = range(0, len(sentences), eff_bs)
        iterator = (
            tqdm(
                indices,
                desc=f"Batches ({task_name})",
                dynamic_ncols=True,
                mininterval=0.2,
                file=sys.stdout,
            )
            if show_pbar
            else indices
        )

        use_dp = (self.device.type == "cuda") and (self.gpu_count > 1)

        for i in iterator:
            batch = sentences[i : i + eff_bs]

            # 1) 用 parser 產生 CPU 上的 batch 字典（保持和你的 pipeline 一致）
            tokens = self.parser([(tmpl, s) for s in batch])

            # 可選：若 parser 支援，做齊長 padding；跟作者的 DataCollatorWithPadding 類似
            if self.pad_to_multiple_of and "attention_mask" in tokens:
                # 一般由 tokenizer 控制；此處假設 parser 已處理，不再二次 pad
                pass

            # 2) 確保 embed_mask 存在
            if "embed_mask" not in tokens:
                tokens["embed_mask"] = tokens.get("attention_mask", None)

            # 3) 移動到裝置
            if self.device.type == "cuda":
                # 多卡：交給 DataParallel 自動 scatter（保持 CPU 張量）
                # 單卡：先把張量 pin_memory 再 non_blocking 搬到 GPU
                if not use_dp:
                    for k, v in list(tokens.items()):
                        if isinstance(v, torch.Tensor):
                            if v.device.type == "cpu":
                                v = v.pin_memory()
                            tokens[k] = v.to(self.device, non_blocking=True)

            # 4) 前向：使用 AMP 提速
            autocast_ctx = (
                autocast_cuda() if self.device.type == "cuda" else nullcontext()
            )

            with autocast_ctx:
                xs = self.model(tokens)  # DataParallel 時自動 scatter/gather

            # 5) pooling 需要 embed_mask；若是單卡已在 GPU，保持 GPU；多卡/CPU 則在 CPU 上處理也可
            xs["embed_mask"] = tokens["embed_mask"]

            xs = self.pooling(xs)  # 內部通常是張量運算，保持一致

            emb: torch.Tensor = xs["sentence_embedding"]

            out_chunks.append(emb.detach().cpu().numpy())

        return np.vstack(out_chunks)
