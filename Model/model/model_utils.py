from peft import LoraConfig, get_peft_model

from model.pytorch_borzoi_model import Borzoi


# model setting up utils
def count_grad_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def set_param_grad(model, trainable_params=[]):

    for name, param in model.named_parameters():
        param.requires_grad = False
        for k in trainable_params:
            if k in name:
                param.requires_grad = True
                break

    return model


def safe_state_dict_loader(org_model_state_dict, load_model_state_dict, partial_load, logger):
    filtered_dict = {}

    for k, v in load_model_state_dict.items():
        if k in org_model_state_dict and v.size() == org_model_state_dict[k].size():
            loaded = False
            for n in partial_load:
                if n in k:
                    filtered_dict[k] = v
                    logger.debug(f"Parameter loaded as specified: {k}.")
                    loaded = True
                    break
            if not loaded:
                logger.debug(f"Parameter not loaded as specified: {k}.")
        else:
            logger.warning(
                f"Parameter not loaded: {k}. This is expected if you have modified the model but not expected if you want to get the exact same model."
            )

    return filtered_dict


def setup_model(config, logger):
    model_config = config.model
    training_config = config.training

    # get our model skeleton
    if "borzoi" in model_config.model_name:
        model_cls = Borzoi
    else:
        logger.error(f"Model {model_config.model_name} is not implemented yet.")
        exit(1)

    model = model_cls.from_hparams(**model_config)

    # initialize the model
    if training_config.load_checkpoint is None:
        model.apply(model._init_weights)

    # in case we need to load the pretrained model and only want to load part of them
    partial_load = model_config.get("partial_load", None)
    partial_load = list(model.state_dict().keys()) if partial_load is None else partial_load

    # if finetune, load the pretrained model
    if training_config.finetune:
        # load the pretrained state dict
        pretrained_model = model_cls.from_pretrained(model_config.model_name)
        org_model_state_dict = model.state_dict()

        updated_model_state_dict = safe_state_dict_loader(
            org_model_state_dict=org_model_state_dict,
            load_model_state_dict=pretrained_model.state_dict(),
            partial_load=partial_load,
            logger=logger,
        )
        org_model_state_dict.update(updated_model_state_dict)
        model.load_state_dict(org_model_state_dict)

        # initialize the finetune model
        if model_config.finetune_method == "lora":
            logger.info("LORA Finetune")
            finetune_config = LoraConfig(**model_config.finetune_param)
            model = get_peft_model(model, finetune_config)
        elif model_config.finetune_method == "finetune_layers":
            logger.info("Finetune the given layers")
            model = set_param_grad(model, **model_config.finetune_param)
        elif model_config.finetune_method == "cont_from_pretrain":
            logger.info("Load from pretrained model and continue train")
            pass
        else:
            logger.error(f"Finetune method {model_config.finetune_method} is not implemented yet.")
            exit(1)

    # get all the trainable parameters
    trainable_params = [n for n, m in model.named_parameters() if m.requires_grad]
    for n in trainable_params:
        logger.debug(f"Trainable module name: {n}")

    trainable_para_num = count_grad_parameters(model)
    total_para_num = count_all_parameters(model)
    logger.info(
        f"Trainable params: {trainable_para_num} || All params: {total_para_num} || Trainable%: {trainable_para_num / total_para_num:.4f}"
    )

    return model
