import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
import torch.nn.functional as F
import os

try:
    from IPython.display import display
    IN_JUPYTER = True
except ImportError:
    display = None
    IN_JUPYTER = False

class KaggleVisCallback(Callback):
    def __init__(self, output_dir='./'):
        super().__init__()
        self.output_dir = output_dir
        self.train_step_losses = []
        self.epoch_train_losses = []
        self.epoch_val_pqs = []
        
        self.cached_val_imgs = []
        self.cached_val_targets = []
        self.cmap = plt.get_cmap('tab20')
        
        # Окошки для обновления в реальном времени (только для Jupyter)
        if IN_JUPYTER:
            try:
                self.step_plot_handle = display(display_id=True)
                self.epoch_plot_handle = display(display_id=True)
                self.img_plot_handle = display(display_id=True)
            except Exception:
                self.step_plot_handle = self.epoch_plot_handle = self.img_plot_handle = None
        else:
            self.step_plot_handle = self.epoch_plot_handle = self.img_plot_handle = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Безопасное извлечение лосса
        if isinstance(outputs, torch.Tensor):
            loss = outputs.item()
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].item()
        else:
            loss = 0.0
            
        self.train_step_losses.append(loss)

        if trainer.global_step > 0 and trainer.global_step % 100 == 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.train_step_losses, label='Train Loss (Step)', color='blue')
            ax.set_title(f'Learning Curve (Step {trainer.global_step})')
            ax.set_xlabel('Global Step')
            ax.set_ylabel('Loss')
            ax.grid(True)
            
            plt.savefig(os.path.join(self.output_dir, 'learning_curve_step.png'), dpi=150, bbox_inches='tight')
            if self.step_plot_handle:
                self.step_plot_handle.update(fig)
            plt.close(fig)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Накапливаем ровно 4 картинки, независимо от batch_size
        if len(self.cached_val_imgs) < 4:
            imgs, targets = batch
            for img, tgt in zip(imgs, targets):
                if len(self.cached_val_imgs) < 4:
                    self.cached_val_imgs.append(img.cpu())
                    self.cached_val_targets.append(tgt)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Считаем средний лосс только если батчи были
        if trainer.num_training_batches > 0 and len(self.train_step_losses) > 0:
            epoch_train_loss = np.mean(self.train_step_losses[-trainer.num_training_batches:])
        else:
            epoch_train_loss = 0.0
            
        self.epoch_train_losses.append(epoch_train_loss)
        
        # Получаем метрику PQ (если ее еще нет в логах, берем 0.0)
        pq_metric = trainer.callback_metrics.get('metrics/val_pq_all', torch.tensor(0.0)).item()
        self.epoch_val_pqs.append(pq_metric)

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        
        ax1.plot(self.epoch_train_losses, 'b-o', label='Train Loss')
        ax2.plot(self.epoch_val_pqs, 'g-s', label='Val PQ')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color='b')
        ax2.set_ylabel('Panoptic Quality', color='g')
        plt.title(f'Epoch {trainer.current_epoch} Metrics Overview')
        
        plt.savefig(os.path.join(self.output_dir, 'epoch_metrics.png'), dpi=150, bbox_inches='tight')
        if self.epoch_plot_handle:
            self.epoch_plot_handle.update(fig)
        plt.close(fig)

        if len(self.cached_val_imgs) > 0:
            self._visualize_predictions(pl_module, trainer.current_epoch, pq_metric)

    def _visualize_predictions(self, pl_module, epoch, pq_metric):
        n_imgs = len(self.cached_val_imgs)
        fig, axes = plt.subplots(2, n_imgs, figsize=(5 * n_imgs, 10))
        if n_imgs == 1: axes = axes[:, None]
        fig.suptitle(f'Epoch: {epoch} | Val PQ achieved: {pq_metric*100:.2f}%', fontsize=16, fontweight='bold')

        # Формируем батч для модели
        batch_imgs = [img.to(pl_module.device) for img in self.cached_val_imgs]
        
        with torch.no_grad():
            transformed_imgs = pl_module.resize_and_pad_imgs_instance_panoptic(batch_imgs)
            mask_logits_per_layer, class_logits_per_layer = pl_module(transformed_imgs)

            mask_logits = mask_logits_per_layer[-1]
            class_logits = class_logits_per_layer[-1]
            
            # Возвращаем размер к исходному (для простоты берем размер первой картинки)
            img_size = self.cached_val_imgs[0].shape[-2:]
            mask_logits = F.interpolate(mask_logits, img_size, mode="bilinear")

            preds = pl_module.to_per_pixel_preds_panoptic(
                mask_logits, class_logits, pl_module.stuff_classes,
                pl_module.mask_thresh, pl_module.overlap_thresh
            )

        for i in range(n_imgs):
            img = self.cached_val_imgs[i]
            target = self.cached_val_targets[i]
            pred_sem = preds[i]["semantic"].cpu()

            # Ground Truth
            gt_sem = torch.zeros(img_size, dtype=torch.long)
            for mask, label in zip(target['masks'], target['labels']):
                gt_sem[mask > 0] = label.cpu()

            # Денормализация
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_np = np.clip(img_np, 0, 1)

            axes[0, i].imshow(img_np)
            axes[0, i].imshow(gt_sem, alpha=0.5, cmap=self.cmap)
            axes[0, i].axis('off')
            if i == 0: axes[0, i].set_title('Ground Truth', fontweight='bold')

            axes[1, i].imshow(img_np)
            axes[1, i].imshow(pred_sem, alpha=0.5, cmap=self.cmap)
            axes[1, i].axis('off')
            if i == 0: axes[1, i].set_title('Model Prediction', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'val_predictions.png'), dpi=200, bbox_inches='tight')
        
        if self.img_plot_handle:
            self.img_plot_handle.update(fig)
        plt.close(fig)