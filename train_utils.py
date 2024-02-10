from data import get_ADE20k, get_ImageNet, get_SAM_dataset
from model import SAM_model

def get_dataset_loaders(config):
    name=config["dataset_name"]
    if name=="ADE20k":
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
    elif name=="SAM_dataset" :
        train_loader = get_SAM_dataset(config["dataset_dir"], config["batch_size"], config["size"], config["num_workers"], config["shuffle"], config["subset_indices"])
        return train_loader
    else:
        raise NotImplementedError()
    print("train size:", len(train_loader))
    print("val size:", len(val_loader))
    return train_loader, val_loader,train_set

def get_scheduler(config, optimizer):
    # get the learning rate scheduler
    name=config["lr_scheduler"]
    elif "ExponentialLR" == name :
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["gamma"])
    else :
        raise NotImplementedError()

def get_loss_fun(config):
    train_crop_size=config["train_crop_size"]
    ignore_value=config["ignore_value"]

    if isinstance(train_crop_size,int):
        crop_h,crop_w=train_crop_size,train_crop_size
    else:
        crop_h,crop_w=train_crop_size

    loss_type="cross_entropy"
    if "loss_type" in config:
        loss_type=config["loss_type"]

    if loss_type=="cross_entropy":
        loss_fun=torch.nn.CrossEntropyLoss(ignore_index=ignore_value)
    elif loss_type=="MSELoss" :
        loss_fun = torch.nn.MSELoss(reduction=config['reduction'])
    elif loss_type=="bootstrapped":
        # 8*768*768/16
        minK=int(config["batch_size"]*crop_h*crop_w/16)
        print(f"bootstrapped minK: {minK}")
        loss_fun=BootstrappedCE(minK,0.3,ignore_index=ignore_value)
    else:
        raise NotImplementedError()
    return loss_fun

def get_optimizer(model,config):

    # Set weight decay and optimizer parameters
    if not config["bn_weight_decay"]:
        p_bn = [p for n, p in model.named_parameters() if "bn" in n]
        p_non_bn = [p for n, p in model.named_parameters() if "bn" not in n]
        optim_params = [
            {"params": p_bn, "weight_decay": 0},
            {"params": p_non_bn, "weight_decay": config["weight_decay"]},
        ]
    else:
        optim_params = model.parameters()

    # Select optimizer
    config_optim = config["optim"]
    if config_optim == "adam" :
        return torch.optim.Adam(optim_params, lr=config["lr"], betas=(config["beta1"], config["beta2"]) , weight_decay=config["weight_decay"])
    elif config_optim == "adamW" :
        return torch.optim.AdamW(optim_params, lr=config["lr"], betas=(config["beta1"], config["beta2"]) , weight_decay=config["weight_decay"])
    else :
        return torch.optim.SGD(
            optim_params,
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
        )

def get_model(name, model_type, pretrained="") :

    if model_type == "sam" :
        model = SAM_model(name, model_type, pretrained)
    
    return model