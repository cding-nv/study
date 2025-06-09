commit 67f7d0e75e39ecf1f775fe317385f5277c532658
Author: root <root@u48a01456.cloud.sqa.nt12>
Date:   Wed Apr 20 04:51:04 2022 +0000

    fused adam, mask_softmax_dropout, dense_gelu_dense, layernorm

diff --git a/main.py b/main.py
old mode 100644
new mode 100755
index ef7bdee..aa4b241
--- a/main.py
+++ b/main.py
@@ -27,6 +27,8 @@ from optimizer import build_optimizer
 from logger import create_logger
 from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
 
+#from apex.contrib.sparsity import ASP
+
 try:
     # noinspection PyUnresolvedReferences
     from apex import amp
@@ -81,12 +83,14 @@ def main(config):
 
     logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
     model = build_model(config)
+    #model = torch.jit.trace(model)
     model.cuda()
     logger.info(str(model))
 
     optimizer = build_optimizer(config, model)
     if config.AMP_OPT_LEVEL != "O0":
         model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
+    #ASP.prune_trained_model(model, optimizer)
     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
     model_without_ddp = model.module
 
@@ -142,6 +146,7 @@ def main(config):
         data_loader_train.sampler.set_epoch(epoch)
 
         train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
+        #break
         if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
             save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
 
@@ -232,6 +237,8 @@ def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mix
                 f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                 f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                 f'mem {memory_used:.0f}MB')
+        #if idx == 3:
+            #break
     epoch_time = time.time() - start
     logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
 
diff --git a/models/build.py b/models/build.py
index 8cefcf8..31b8c6b 100644
--- a/models/build.py
+++ b/models/build.py
@@ -7,7 +7,7 @@
 
 from .swin_transformer import SwinTransformer
 from .swin_mlp import SwinMLP
-
+import apex.normalization
 
 def build_model(config):
     model_type = config.MODEL.TYPE
@@ -25,6 +25,7 @@ def build_model(config):
                                 qk_scale=config.MODEL.SWIN.QK_SCALE,
                                 drop_rate=config.MODEL.DROP_RATE,
                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
+                                norm_layer=apex.normalization.FusedLayerNorm,
                                 ape=config.MODEL.SWIN.APE,
                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)
diff --git a/models/swin_transformer.py b/models/swin_transformer.py
old mode 100644
new mode 100755
index cfeb0f2..e1790cf
--- a/models/swin_transformer.py
+++ b/models/swin_transformer.py
@@ -9,23 +9,32 @@ import torch
 import torch.nn as nn
 import torch.utils.checkpoint as checkpoint
 from timm.models.layers import DropPath, to_2tuple, trunc_normal_
+from apex.contrib.multihead_attn import fast_mask_softmax_dropout_func
 
+from apex.fused_dense import FusedDenseGeluDense
+
+import pdb
 
 class Mlp(nn.Module):
     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
         super().__init__()
         out_features = out_features or in_features
         hidden_features = hidden_features or in_features
-        self.fc1 = nn.Linear(in_features, hidden_features)
-        self.act = act_layer()
-        self.fc2 = nn.Linear(hidden_features, out_features)
+        #self.fc1 = nn.Linear(in_features, hidden_features)
+        #self.act = act_layer()
+        #self.fc2 = nn.Linear(hidden_features, out_features)
+
+        self.fused_d_g_d = FusedDenseGeluDense(in_features, hidden_features, out_features)
+
         self.drop = nn.Dropout(drop)
 
     def forward(self, x):
-        x = self.fc1(x)
-        x = self.act(x)
-        x = self.drop(x)
-        x = self.fc2(x)
+        #x = self.fc1(x)
+        #x = self.act(x)
+        #x = self.drop(x)
+        #x = self.fc2(x)
+        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
+        x = self.fused_d_g_d(x)
         x = self.drop(x)
         return x
 
@@ -86,28 +95,28 @@ class WindowAttention(nn.Module):
         self.scale = qk_scale or head_dim ** -0.5
 
         # define a parameter table of relative position bias
-        self.relative_position_bias_table = nn.Parameter(
-            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
+        #self.relative_position_bias_table = nn.Parameter(
+        #    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
 
         # get pair-wise relative position index for each token inside the window
-        coords_h = torch.arange(self.window_size[0])
-        coords_w = torch.arange(self.window_size[1])
-        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
-        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
-        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
-        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
-        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
-        relative_coords[:, :, 1] += self.window_size[1] - 1
-        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
-        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
-        self.register_buffer("relative_position_index", relative_position_index)
+        #coords_h = torch.arange(self.window_size[0])
+        #coords_w = torch.arange(self.window_size[1])
+        #coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
+        #coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
+        #relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
+        #relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
+        #relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
+        #relative_coords[:, :, 1] += self.window_size[1] - 1
+        #relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
+        #relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
+        #self.register_buffer("relative_position_index", relative_position_index)
 
         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
         self.attn_drop = nn.Dropout(attn_drop)
         self.proj = nn.Linear(dim, dim)
         self.proj_drop = nn.Dropout(proj_drop)
 
-        trunc_normal_(self.relative_position_bias_table, std=.02)
+        #trunc_normal_(self.relative_position_bias_table, std=.02)
         self.softmax = nn.Softmax(dim=-1)
 
     def forward(self, x, mask=None):
@@ -116,31 +125,44 @@ class WindowAttention(nn.Module):
             x: input features with shape of (num_windows*B, N, C)
             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
         """
+        #torch.cuda.nvtx.mark("attn_start")
         B_, N, C = x.shape
         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
 
+        #pdb.set_trace()
+
         q = q * self.scale
         attn = (q @ k.transpose(-2, -1))
 
-        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
-            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
-        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
-        attn = attn + relative_position_bias.unsqueeze(0)
-
-        if mask is not None:
-            nW = mask.shape[0]
-            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
-            attn = attn.view(-1, self.num_heads, N, N)
-            attn = self.softmax(attn)
-        else:
-            attn = self.softmax(attn)
-
-        attn = self.attn_drop(attn)
-
+        #relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
+        #    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
+        #relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
+        #attn = attn + relative_position_bias.unsqueeze(0)
+
+        # if mask is not None:
+        #     nW = mask.shape[0]
+        #     #attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
+        #     #attn = attn.view(-1, self.num_heads, N, N)
+        #     #attn = self.softmax(attn)
+        #     #attn = self.attn_drop(attn)
+        #     attn = attn.view(-1, N, N)
+        #     mask = mask.unsqueeze(1).unsqueeze(0)
+        #     mask = mask.view(-1, 49)
+        #     attn = fast_mask_softmax_dropout_func(True, self.num_heads, attn, mask, True, 0.0)
+        # else:
+        #     #attn = self.softmax(attn)
+        #     #attn = self.attn_drop(attn)
+        #     attn = attn.view(-1, N, N)
+        #     attn = fast_mask_softmax_dropout_func(True, self.num_heads, attn, mask, False, 0.0)
+
+        attn = attn.view(-1, N, N)
+        attn = fast_mask_softmax_dropout_func(True, self.num_heads, attn, None, False, 0.0)
+        attn = attn.view(-1, self.num_heads, N, N)
         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
         x = self.proj(x)
         x = self.proj_drop(x)
+        #torch.cuda.nvtx.mark("attn_end")
         return x
 
     def extra_repr(self) -> str:
@@ -203,6 +225,7 @@ class SwinTransformerBlock(nn.Module):
         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
         self.norm2 = norm_layer(dim)
         mlp_hidden_dim = int(dim * mlp_ratio)
+        #pdb.set_trace()
         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
 
         if self.shift_size > 0:
@@ -250,7 +273,9 @@ class SwinTransformerBlock(nn.Module):
         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
 
         # W-MSA/SW-MSA
+        torch.cuda.nvtx.range_push("attn_start")
         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
+        torch.cuda.nvtx.range_pop()
 
         # merge windows
         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
@@ -265,8 +290,13 @@ class SwinTransformerBlock(nn.Module):
 
         # FFN
         x = shortcut + self.drop_path(x)
-        x = x + self.drop_path(self.mlp(self.norm2(x)))
-
+        #pdb.set_trace()
+        x_norm2 = self.norm2(x)
+        x_mlp = self.mlp(x_norm2)
+        x_mlp = x_mlp.view(x.shape[0], x.shape[1], x.shape[2])
+        #x_drop_path = self.drop_path(x_mlp)
+        #x = x + self.drop_path(self.mlp(self.norm2(x)))
+        x = x + self.drop_path(x_mlp)
         return x
 
     def extra_repr(self) -> str:
diff --git a/optimizer.py b/optimizer.py
index 3c57ce0..9fa530a 100644
--- a/optimizer.py
+++ b/optimizer.py
@@ -6,7 +6,7 @@
 # --------------------------------------------------------
 
 from torch import optim as optim
-
+from apex.optimizers import FusedAdam
 
 def build_optimizer(config, model):
     """
@@ -26,9 +26,9 @@ def build_optimizer(config, model):
         optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                               lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
     elif opt_lower == 'adamw':
-        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
-                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
-
+        #optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
+        #                        lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
+        optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
     return optimizer
 
 
