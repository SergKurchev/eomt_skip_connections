# tests/test_kaggle_vis_callback.py
import os
import sys
import pytest
import torch
import lightning.pytorch as L
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.kaggle_vis_callback import KaggleVisCallback

class DummyDataset(Dataset):
    def __len__(self): return 150
    def __getitem__(self, idx):
        img = torch.rand(3, 128, 128)
        target = {
            'masks': torch.randint(0, 2, (2, 128, 128), dtype=torch.bool),
            'labels': torch.tensor([1, 2])
        }
        return img, target

def dummy_collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return imgs, targets

class DummyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.stuff_classes = [0]
        self.mask_thresh = 0.5
        self.overlap_thresh = 0.5

    def forward(self, x):
        batch_size = len(x)
        mask_logits = [torch.randn(batch_size, 5, 32, 32)]
        class_logits = [torch.randn(batch_size, 5, 3)]
        return mask_logits, class_logits

    def training_step(self, batch, batch_idx):
        loss = torch.rand(1, requires_grad=True) * 10 / (self.current_epoch + 1)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        pq = 0.3 + (self.current_epoch * 0.1)
        self.log('metrics/val_pq_all', pq, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        return imgs
        
    def to_per_pixel_preds_panoptic(self, mask_logits, class_logits, stuff, mask_th, ov_th):
        batch_size = mask_logits.shape[0]
        img_size = mask_logits.shape[-2:]
        return [{"semantic": torch.randint(0, 3, img_size)} for _ in range(batch_size)]

def test_kaggle_vis_callback(tmp_path):

    output_dir = str(tmp_path)
    # output_dir = "./tests/vis/"
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    model = DummyModel()
    train_loader = DataLoader(DummyDataset(), batch_size=2, collate_fn=dummy_collate)
    val_loader = DataLoader(DummyDataset(), batch_size=1, collate_fn=dummy_collate)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='best-mock-{epoch:02d}',
            monitor='metrics/val_pq_all',
            mode='max',
            save_top_k=1
        ),
        KaggleVisCallback(output_dir=output_dir)
    ]

    trainer = L.Trainer(
        max_epochs=2, 
        limit_train_batches=150, 
        limit_val_batches=5,
        callbacks=callbacks,
        accelerator='cpu',
        enable_model_summary=False,
        logger=False 
    )

    trainer.fit(model, train_loader, val_loader)
    
    assert os.path.exists(os.path.join(output_dir, 'learning_curve_step.png')), "График 'learning_curve_step.png' не был создан"
    assert os.path.exists(os.path.join(output_dir, 'epoch_metrics.png')), "График 'epoch_metrics.png' не был создан"
    assert os.path.exists(os.path.join(output_dir, 'val_predictions.png')), "График 'val_predictions.png' не был создан"
    
    saved_ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    assert len(saved_ckpts) == 1, f"Ожидался 1 чекпоинт, найдено: {len(saved_ckpts)}"