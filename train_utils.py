from data import get_ADE20k


def get_dataset_loaders(config):
    name=config["dataset_name"]
    if name=="cityscapes":
        train_loader, val_loader,train_set=get_ADE20k(
            config["dataset_dir"],
            config["batch_size"],
            config["train_min_size"],
            config["train_max_size"],
            config["train_crop_size"],
            config["val_input_size"],
            config["val_label_size"],
            config["aug_mode"],
            config["class_uniform_pct"],
            config["train_split"],
            config["val_split"],
            config["num_workers"],
            config["ignore_value"]
        )
    else:
        raise NotImplementedError()
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader,train_set