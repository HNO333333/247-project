import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from typing import ClassVar, Sequence, Any
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from transformers import WhisperConfig
from hydra.utils import instantiate
from torchmetrics import MetricCollection
import logging
from einops.layers.torch import Rearrange

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

from emg2qwerty.lightning import WindowedEMGDataModule
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm
from emg2qwerty.customs.data import get_custom_collate
from emg2qwerty.customs.module_whisper import WhisperEncoder
from emg2qwerty.customs.module_conformer import Conformer
from emg2qwerty.charset import charset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.data import LabelData
from emg2qwerty import utils

logger = logging.getLogger(__name__)


class WhisperWindowedEMGDataModule(WindowedEMGDataModule):
    """
    Custom DataModule that uses a custom collate function: padding to the same length for each batch (for whisper)
    """

    def __init__(self, max_T: int = 622, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_T = max_T

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=get_custom_collate(self.max_T),
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_custom_collate(self.max_T),
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_custom_collate(self.max_T, is_test=True),
            pin_memory=True,
            persistent_workers=True,
        )


class WhisperEncoderModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        max_T: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.max_T = max_T
        whisper_config = WhisperConfig.from_pretrained(
            "./emg2qwerty/customs/whisper_config/whisper_config.json"
        )

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            WhisperEncoder(whisper_config),
            nn.Linear(whisper_config.d_model, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]  # [T, N, n_bands, n_channels, n_freqs]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size
        num_freqs = inputs.shape[-1]
        T = inputs.shape[0]

        if phase != "test":
            emissions = self.forward(inputs)
        else:
            last_padding = inputs.shape[0] % self.max_T
            if last_padding == 0:
                inputs = inputs.reshape(
                    self.max_T, -1, self.NUM_BANDS, self.ELECTRODE_CHANNELS, num_freqs
                )
            else:
                n_samples = inputs.shape[0] // self.max_T + 1
                inputs_ = torch.zeros(
                    self.max_T,
                    n_samples,
                    self.NUM_BANDS,
                    self.ELECTRODE_CHANNELS,
                    num_freqs,
                    device=inputs.device,
                )
                inputs_[:, :-1, :, :, :] = inputs[:-last_padding, :, :, :, :].reshape(
                    self.max_T, -1, self.NUM_BANDS, self.ELECTRODE_CHANNELS, num_freqs
                )
                inputs_[:last_padding, -1, :, :, :] = inputs[
                    -last_padding:, :, :, :, :
                ].squeeze(1)
                inputs = inputs_
            emissions = self.forward(inputs)
            emissions = emissions.reshape(-1, 1, charset().num_classes)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = T - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

# --- Conformer ---

class ConformerModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        emb_size: int, # default: 80
        depth: int, # default: 6
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # raw input Model
        # self.model = nn.Sequential(
        #     # inputs: (time, batch_size, conv_channel, electrode_channels)
        #     # Rearrange("t n c e -> n c e t"),
            
        #     # inputs: (time, batch_size, num_bands, electrode_channels)
        #     MultiBandRotationInvariantMLP(
        #         in_features=self.ELECTRODE_CHANNELS,
        #         mlp_features=[16, 16],
        #         num_bands=self.NUM_BANDS,
        #     ),
        #     Rearrange("t n nb ec -> n 1 (nb ec) t"),
        #     Conformer(emb_size=emb_size, depth=depth, n_classes=charset().num_classes),
        #     Rearrange("n t e -> t n e"),
        #     nn.LogSoftmax(dim=-1),
        # )

        # spectrogram input Model
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            Rearrange("t n f -> n 1 f t"),
            Conformer(emb_size=emb_size, depth=depth, n_classes=charset().num_classes),
            Rearrange("n t e -> t n e"),
            nn.LogSoftmax(dim=-1),
        )


        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        downsample_rate = 15
        T_diff = inputs.shape[0] // downsample_rate - emissions.shape[0]
        emission_lengths = input_lengths // downsample_rate - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )