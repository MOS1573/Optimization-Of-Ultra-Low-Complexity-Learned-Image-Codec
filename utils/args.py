import argparse
import yaml


def get_args(config_stage = "train"):
    with open("configs/general/general.yaml", "r") as f:
        config = yaml.safe_load(f)
        general_config = config.get("general", {})
        config = config.get(config_stage, {})
        config.update(general_config)
    return config

        



def train_options():
    parser = argparse.ArgumentParser(description="Training script.")

    # Load all arguments from the config file
    config = get_args("train")


    for key, value in config.items():
        arg_type = type(value)
        if arg_type == list:
            arg_type = type(value[0])
        
        parser.add_argument(
            f"--{key}",
            default=value,
            type=arg_type,
            required=False,
            help=f"{key} (default: %(default)s)"
        )

    args = parser.parse_args()
    return args


def test_options():
    parser = argparse.ArgumentParser(description="Testing script.")

    # Load all arguments from the config file
    config = get_args("test")

    for key, value in config.items():
        arg_type = type(value)
        if arg_type == list:
            arg_type = type(value[0])

        parser.add_argument(
            f"--{key}",
            default=value,
            type=arg_type,
            required=False,
            help=f"{key} (default: %(default)s)"
        )

    args = parser.parse_args()
    return args
