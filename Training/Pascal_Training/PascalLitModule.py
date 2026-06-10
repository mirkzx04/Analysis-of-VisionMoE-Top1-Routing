import pytorch_lightning as pl

import torch

from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAccuracy

class PascalLitModule(pl.LightningModule): 
    def __init__(
        self, 
        model, 
        router_lr, 
        head_lr, 
        weight_decay, 
        num_classes, 
        train_epochs, 
        temp_init, 
        temp_final, 
        temp_epochs 
    ):

        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.lr_router = lr_router 
        self.head_lr = head_lr 

        self.temp_init = temp_init 
        self.temp_final = temp_final 
        self.temp_epochs = temp_epochs 

        self.head_warmup = 10
        self.router_warmup = 5 

        self.train_epochs = train_epochs 

        # Collect training losses 
        self.train_seg_loosses = []
        self.train_aux_losses = []
        self.train_raw_spatial_losses = []
        self.train_weighted_spatial_losses = []
        self.train_total_losses = []

        # Collect validation losses 
        self.val_class_losses = []
        self.val_aux_losses = []
        self.val_raw_spatial_losses = []
        self.val_weighted_spatial_losses = []
        self.val_total_losses = []

        self.gradient_norm_router = []
        self.gradient_norm_backbone = []
        
        self.z_loss_weigth = 0.0 
        self.spatial_loss_weight = 0.0

        # define loss function 
        self.train_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.val_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

        # define accuracy metrics 
        accuracy_metrics = {
            "miou_val" : JaccardIndex(task = "multiclass", num_classes = num_classes, ignore_index = 255, average = "macro").to(device),
            "pixacc_val" : MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average="micro").to(device),
            "macc_val" : MulticlassAccuracy(num_classes=num_classes, ignore_index=255, average="macro").to(device)
        }
    
    def forward(self, x): 
        return self.model(x, current_epoch=self.current_epoch, collect_routes=True, collect_debug=False)

    def training_setp(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        # Execute forward pass
        logits, spatial_loss, z_loss, _ = self(data)

        # Compute loss 
        seg_loss = self.train_loss(logits, label)

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        aux_loss = (z_loss * self.z_loss_weigth) + weighted_spatial_loss 
        total_loss = aux_loss + seg_loss

        self.train_class_losses.append(class_loss.detach())
        self.train_aux_losses.append(aux_loss.detach())
        self.train_raw_spatial_losses.append(spatial_loss.detach())
        self.train_weighted_spatial_losses.append(weighted_spatial_loss.detach())

        return {"loss" : total_loss}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.to(self.device), label.to(self.device)

        # Execute forward pass
        logits, spatial_loss, z_loss, _ = self(data)

        # Compute loss 
        seg_loss = self.val_loss(logits, label)

        weighted_spatial_loss = spatial_loss * self.spatial_loss_weight
        aux_loss = (z_loss * self.z_loss_weigth) + weighted_spatial_loss 
        total_loss = aux_loss + seg_loss

        self.val_class_losses.append(class_loss.detach())
        self.val_aux_losses.append(aux_loss.detach())
        self.val_raw_spatial_losses.append(spatial_loss.detach())
        self.val_weighted_spatial_losses.append(weighted_spatial_loss.detach())
    




