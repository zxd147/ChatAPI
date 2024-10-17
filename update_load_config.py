import yaml
import json


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(file)
        elif config_path.endswith('.json'):
            config = json.load(file)
        else:
            raise ValueError("Unsupported config file format. Please use .yaml, .yml or .json")
    return config


def update_config(config_path, new_config):
    # 加载现有配置
    with open(config_path, 'r', encoding='utf-8') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(file)
        elif config_path.endswith('.json'):
            config = json.load(file)
        else:
            raise ValueError("Unsupported config file format. Please use .yaml, .yml or .json")

    # 更新配置
    config.update(new_config)

    # 写回配置文件
    with open(config_path, 'w', encoding='utf-8') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.safe_dump(config, file, allow_unicode=True)
        elif config_path.endswith('.json'):
            json.dump(config, file, ensure_ascii=False, indent=4)
        else:
            raise ValueError("Unsupported config file format. Please use .yaml, .yml or .json")


