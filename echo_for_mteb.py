# echo_for_mteb.py
from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser
from mteb.encoder_interface import PromptType
from tqdm.auto import tqdm
import sys, os
import torch
import numpy as np


class EchoModel:
    def __init__(
        self, path_to_model, templates, max_length=300, pooling_strategy="mean"
    ):
        self.templates = templates
        self.model = EchoEmbeddingsMistral.from_pretrained(path_to_model).eval()
        self.parser = EchoParser(path_to_model, templates, max_length=max_length)
        self.pooling = EchoPooling(strategy=pooling_strategy)

        # device 與多 GPU
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            # DataParallel 會自動複製到所有可用 GPU；主卡預設 cuda:0
            self.model = torch.nn.DataParallel(self.model)
            self.device = torch.device("cuda:0")
        elif n_gpu == 1:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

        os.environ.setdefault("TQDM_DISABLE", "0")

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

    @torch.no_grad()
    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        if not sentences:
            return np.empty((0, 0), dtype=np.float32)

        batch_size: int = int(kwargs.get("batch_size", 128))
        show_pbar: bool = bool(kwargs.get("show_progress_bar", False))

        tmpl = self._template_key(prompt_type)
        out_chunks = []

        indices = range(0, len(sentences), batch_size)
        iterator = (
            tqdm(
                indices,
                desc=f"Batches ({task_name})",
                dynamic_ncols=True,
                mininterval=0.2,
                file=sys.stdout,
                leave=False,
            )
            if show_pbar
            else indices
        )

        for i in iterator:
            batch = sentences[i : i + batch_size]

            # 1) 用 parser 產生 CPU 上的 batch 字典
            tokens = self.parser([(tmpl, s) for s in batch])

            # 2) 確保 embed_mask 一起走（若沒有就用 attention_mask 代替或自行定義）
            if "embed_mask" not in tokens:
                tokens["embed_mask"] = tokens.get("attention_mask", None)

            # 3) 關鍵：傳「單一 dict」給 model。讓 DataParallel 自動 scatter/gather
            xs = self.model(tokens)

            # 4) pooling 需要 embed_mask，把它補進去（此時 xs 已經被 DP 聚合到主卡）
            xs["embed_mask"] = tokens["embed_mask"]

            # 5) 後處理
            xs = self.pooling(xs)
            emb = xs["sentence_embedding"].detach().cpu().numpy().astype(np.float32)
            out_chunks.append(emb)

        return np.vstack(out_chunks)
