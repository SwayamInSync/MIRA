import json


class Config:
    def __init__(self, camera_matrix_dim, camera_embed_dim, decoder_hidden_dim, num_layers, num_heads,
                 weight_decay, learning_rate, batch_size, triplane_feat_res, triplane_res, triplane_dim,
                 rendering_samples_per_ray, render_size, source_size, lpips_net, num_epochs, save_every_epoch,
                 focal_length, principal_point, supervision_k, model_preloading_strategy, model_save_path, data_dir):
        self.camera_matrix_dim = camera_matrix_dim
        self.camera_embed_dim = camera_embed_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.triplane_feat_res = triplane_feat_res
        self.triplane_res = triplane_res
        self.triplane_dim = triplane_dim
        self.rendering_samples_per_ray=rendering_samples_per_ray
        self.render_size = render_size
        self.source_size = source_size
        self.lpips_net = lpips_net
        self.num_epochs = num_epochs
        self.save_every_epoch = save_every_epoch
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.supervision_k = supervision_k
        self.model_preloading_strategy = model_preloading_strategy
        self.model_save_path = model_save_path
        self.data_dir = data_dir


    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as file:
            config_dict = json.load(file)
        return cls(**config_dict)

    def to_dict(self):
        return vars(self)

    def __str__(self):
        string = ""
        for k, v in self.to_dict().items():
            string += f"{k}: {v}\n"
        return string


if __name__ == "__main__":
    config = Config.from_json("final_train_config.json")
    print(config)
