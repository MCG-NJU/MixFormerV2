from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mixformer2_vit_online.config import update_config_from_file, cfg


def parameters(yaml_name: str, model=None, search_area_scale=None, online_size=None, update_interval=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/mixformer2_vit_online/{}.yaml'.format(yaml_name))
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    if search_area_scale is not None:
        params.search_factor = search_area_scale
    else:
        params.search_factor = cfg.TEST.SEARCH_FACTOR
    if online_size is not None:
        params.online_size = online_size
    if update_interval is not None:
        params.update_interval = update_interval
    # params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    if model is None:
        raise NotImplementedError("Please set proper model to test.")
    else:
        params.checkpoint = os.path.join(save_dir, model)

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
