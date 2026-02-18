import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import sys

class SimpleViT(nn.Module):
    def __init__(self, img_size, patch_size=16, embed_dim=384, depth=6, heads=6):
        super().__init__()
        self.patch_size = patch_size
        W, H = img_size
        
        self.h_grid = H // patch_size
        self.w_grid = W // patch_size
        num_patches = self.h_grid * self.w_grid
        
        self.input_query = nn.Parameter(torch.randn(1, 3, H, W))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, 
                                             batch_first=True, norm_first=True,
                                             dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, patch_size**2 * 3)

    def forward(self):
        x = self.input_query
        
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        x = self.head(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.h_grid, self.w_grid)
        return torch.sigmoid(nn.functional.pixel_shuffle(x, self.patch_size))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print("config.json not found")
        sys.exit(1)
        


    model = SimpleViT(
        img_size=(cfg["img_width"], cfg["img_height"]),
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        heads=cfg["heads"]
    ).to(device)


    try:
        state_dict = torch.load("overfitted_vit.pth", map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully")
    except FileNotFoundError:
        print("You have forgotten the weights")
        sys.exit(1)

    model.eval()
    with torch.no_grad():
        output_tensor = model()
        

        output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1)
        output_np = (output_tensor.numpy() * 255).astype(np.uint8)
        
        result_img = Image.fromarray(output_np)
        result_img.save("restored_image.jpg")
        print("Image saved to './restored_image.jpg'")

if __name__ == "__main__":
    main()
