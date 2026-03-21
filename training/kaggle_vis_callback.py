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
    def __init__(self, output_dir='./result_graphs'):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.train_step_losses = []
        self.epoch_train_losses = []
        self.epoch_val_pqs = []
        
        self.cached_val_imgs = []
        self.cached_val_targets = []
        
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
        if not trainer.is_global_zero: return

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
            if self.step_plot_handle: self.step_plot_handle.update(fig)
            plt.close(fig)

    def on_validation_epoch_start(self, trainer, pl_module):
        # ЗАЩИТА: Только главная видеокарта и только после Sanity Check
        if trainer.sanity_checking or not trainer.is_global_zero: 
            return

        # Если мы еще не закэшировали картинки (на первой реальной эпохе)
        if len(self.cached_val_imgs) == 0:
            # Стучимся напрямую в датасет, минуя батчи и шаффлинг DDP
            val_dataloader = trainer.val_dataloaders
            if isinstance(val_dataloader, list):
                val_dataloader = val_dataloader[0]
            dataset = val_dataloader.dataset
            
            # Строго берем индексы 0, 1, 2, 3
            for i in range(min(4, len(dataset))):
                img, tgt = dataset[i]
                self.cached_val_imgs.append(img.cpu())
                
                # Безопасно переносим таргеты на CPU
                tgt_cpu = {}
                for k, v in tgt.items():
                    if isinstance(v, torch.Tensor):
                        tgt_cpu[k] = v.cpu()
                    else:
                        tgt_cpu[k] = v
                self.cached_val_targets.append(tgt_cpu)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero: 
            return

        if trainer.num_training_batches > 0 and len(self.train_step_losses) > 0:
            epoch_train_loss = np.mean(self.train_step_losses[-trainer.num_training_batches:])
        else:
            epoch_train_loss = 0.0
            
        self.epoch_train_losses.append(epoch_train_loss)
        
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
        if self.epoch_plot_handle: self.epoch_plot_handle.update(fig)
        plt.close(fig)

        if len(self.cached_val_imgs) > 0:
            self._visualize_predictions(pl_module, trainer.current_epoch, pq_metric)
            # Внимание: мы БОЛЬШЕ НЕ ОЧИЩАЕМ кэш здесь!
            # Эти 4 картинки останутся в памяти (это копейки МБ) на все 24 эпохи.

    @staticmethod
    def create_overlay(img_np, sem, inst, mapping, alpha=0.7):
        """Накладывает цветную маску и черные границы поверх оригинального изображения"""
        h, w = sem.shape
        out = img_np.copy()
        
        for s in np.unique(sem):
            color = mapping.get(s)
            if color is not None:
                mask = (sem == s)
                out[mask] = out[mask] * (1 - alpha) + np.array(color) * alpha

        combined = sem.astype(np.int64) * 100000 + inst.astype(np.int64)
        border = np.zeros((h, w), dtype=bool)
        border[1:, :] |= combined[1:, :] != combined[:-1, :]
        border[:-1, :] |= combined[1:, :] != combined[:-1, :]
        border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
        border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
        
        out[border] = [0, 0, 0]
        return out

    def _visualize_predictions(self, pl_module, epoch, pq_metric):
        n_imgs = len(self.cached_val_imgs)
        fig, axes = plt.subplots(n_imgs, 3, figsize=(15, 5 * n_imgs))
        if n_imgs == 1: axes = [axes]
        fig.suptitle(f'Epoch: {epoch} | Val PQ achieved: {pq_metric*100:.2f}%', fontsize=16, fontweight='bold')

        batch_imgs = [img.to(pl_module.device) for img in self.cached_val_imgs]
        img_sizes = [img.shape[-2:] for img in batch_imgs] 
        
        with torch.no_grad():
            transformed_imgs = pl_module.resize_and_pad_imgs_instance_panoptic(batch_imgs)
            mask_logits_per_layer, class_logits_per_layer = pl_module(transformed_imgs)

            mask_logits = mask_logits_per_layer[-1]
            class_logits = class_logits_per_layer[-1]
            
            mask_logits = F.interpolate(mask_logits, pl_module.img_size, mode="bilinear")
            mask_logits_list = pl_module.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)

            preds = pl_module.to_per_pixel_preds_panoptic(
                mask_logits_list, class_logits, pl_module.stuff_classes,
                pl_module.mask_thresh, pl_module.overlap_thresh
            )

        for i in range(n_imgs):
            img = self.cached_val_imgs[i]
            target = self.cached_val_targets[i]
            
            pred_np = preds[i].cpu().numpy()
            sem_pred, inst_pred = pred_np[..., 0], pred_np[..., 1]

            target_seg = pl_module.to_per_pixel_targets_panoptic([target])[0].cpu().numpy()
            sem_target, inst_target = target_seg[..., 0], target_seg[..., 1]

            all_ids = np.union1d(np.unique(sem_pred), np.unique(sem_target))
            mapping = {
                s: (
                    None if s == -1 or s == pl_module.num_classes 
                    else plt.cm.hsv(j / max(1, len(all_ids)))[:3]
                )
                for j, s in enumerate(all_ids)
            }

            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = img_np / 255.0
            img_np = np.clip(img_np, 0, 1)

            vis_pred = self.create_overlay(img_np, sem_pred, inst_pred, mapping, alpha=0.7)
            vis_target = self.create_overlay(img_np, sem_target, inst_target, mapping, alpha=0.7)

            axes[i][0].imshow(img_np)
            if i == 0: axes[i][0].set_title('Original Input', fontweight='bold')
            axes[i][0].axis('off')

            axes[i][1].imshow(vis_pred)
            if i == 0: axes[i][1].set_title('Prediction (Alpha 0.7 + Borders)', fontweight='bold')
            axes[i][1].axis('off')

            axes[i][2].imshow(vis_target)
            if i == 0: axes[i][2].set_title('Ground Truth Target', fontweight='bold')
            axes[i][2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'val_predictions.png'), dpi=200, bbox_inches='tight')
        
        if self.img_plot_handle: self.img_plot_handle.update(fig)
        plt.close(fig)