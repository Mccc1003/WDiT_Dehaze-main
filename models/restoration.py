import torch
import numpy as np
import utils
import os
import torch.nn.functional as F


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        self.device = config.device

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.test_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                b, c, img_h, img_w = x.shape
                patch_size = int(self.config.data.patch_size)
                overlap = patch_size // 2  
                pred_x = torch.zeros((b, 3, img_h, img_w), device=self.device)

                count_map = torch.zeros((b, 3, img_h, img_w), device=self.device)  

                for h in range(0, img_h, patch_size - overlap):
                    for w in range(0, img_w, patch_size - overlap):
                        h_end = min(h + patch_size, img_h)
                        w_end = min(w + patch_size, img_w)
                        h_start = h_end - patch_size
                        w_start = w_end - patch_size

                        patch = x[:, :, h_start:h_end, w_start:w_end]
                        out = self.diffusive_restoration(patch.to(self.device))
                        # pred_patch = out["pred_x"]
                        pred_patch = out

                        pred_x[:, :, h_start:h_end, w_start:w_end] += pred_patch[:, :, :h_end-h_start, :w_end-w_start]
                        count_map[:, :, h_start:h_end, w_start:w_end] += 1

                pred_x /= count_map
                utils.logging.save_image(pred_x, os.path.join(image_folder, f"{y[0]}.png"))
                print(f"processing image {y[0]}")

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]


