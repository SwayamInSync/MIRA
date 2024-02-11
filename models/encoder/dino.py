import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel


class DinoEncoder(nn.Module):
    def __init__(self, freeze: bool = True):
        super().__init__()
        self.model = ViTModel.from_pretrained('facebook/dino-vitb16', add_pooling_layer=False)
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        if freeze:
            self._freeze()

    def forward(self, image):
        """
        :param image: [N, C, H, W] (scaled between 0-1)
        :return: Tensor[N, 1025, 768]
        """
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False, do_resize=False).to(
            self.model.device)
        # This resampling of positional embedding uses bicubic interpolation (for different image sizes)
        outputs = self.model(**inputs, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _freeze(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import torch

    encoder = DinoEncoder()
    img = torch.randint(0, 256, (3, 512, 512)).unsqueeze(0) / 255.0
    op = encoder(img)
    assert op.shape == (1, 1025, 768)
