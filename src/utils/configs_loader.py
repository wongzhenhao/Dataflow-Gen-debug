import yaml

def load_config(yaml_path):
    """
    加载 YAML 配置文件。

    Args:
        yaml_path (str): 配置文件路径。

    Returns:
        dict: 配置参数的字典。
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到：{yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"解析 YAML 文件时出错：{e}")
