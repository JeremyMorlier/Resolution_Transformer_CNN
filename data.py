
import torch
from datasets.ADE20k import ADE20k

if __name__ == "__main__" :
    import yaml

    def load_yaml(config_filename) :
        with open(config_filename) as file:
            config=yaml.full_load(file)
        return config
    
    config_filename = "test_config"
    config = load_yaml("configs/" + config_filename + ".yaml")
    root = config["root"]
    split = config["split"]
    dataset = ADE20k(root, split, None)