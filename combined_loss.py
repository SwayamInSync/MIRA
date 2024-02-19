import torch
import lpips
import torch.nn as nn

from config import Config

c = Config.from_json("final_train_config.json")


class ReconstructionLoss(nn.Module):
    def __init__(self, lambda_value=2.0):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        net = getattr(c, "lpips_net", None)
        self.lpips_loss = lpips.LPIPS(net=net if net is not None else "alex")
        self.lambda_value = lambda_value

    def forward(self, predicted_images, target_images):
        mse = self.mse_loss(predicted_images, target_images)
        lpips_value = self.lpips_loss(predicted_images, target_images)
        combined_loss = mse + self.lambda_value * lpips_value
        return torch.mean(combined_loss)


if __name__ == "__main__":
    x,y = torch.rand(4, 3, 128, 128), torch.rand(4, 3, 128, 128)
    loss_fn = ReconstructionLoss()
    loss = loss_fn(x,y)
    print(loss)