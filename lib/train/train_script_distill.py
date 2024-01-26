from copy import deepcopy
import os
# loss function related
from lib.utils.box_ops import ciou_loss, ciou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mixformer_vit import build_mixformer_vit
from lib.models.mixformer2_vit import build_mixformer2_vit, build_mixformer2_vit_stu
# forward propagation related
from lib.train.actors import MixFormerDistillStage1Actor, MixFormerDistillStage2Actor
# for import modules
import importlib


def build_network(script_name, cfg, teacher: bool=False):
    train = (not teacher)
    # Create network
    if script_name == "mixformer_vit":
        net = build_mixformer_vit(cfg, train=train)
    elif script_name == "mixformer2_vit":
        net = build_mixformer2_vit(cfg, train=train)
    elif script_name == "mixformer2_vit_stu":
        net = build_mixformer2_vit_stu(cfg, train=train)
    else:
        raise ValueError("illegal script name: {}.".format(script_name))
    return net


def run(settings):
    settings.description = 'Training script for mixformer distill'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update the default teacher configs with teacher config file
    if not os.path.exists(settings.cfg_file_teacher):
        raise ValueError("%s doesn't exist." % settings.cfg_file_teacher)
    config_module_teacher = importlib.import_module("lib.config.%s.config" % settings.script_teacher)
    cfg_teacher = config_module_teacher.update_new_config_from_file(settings.cfg_file_teacher)
    if settings.local_rank in [-1, 0]:
        print("New teacher configuration is shown below.")
        for key in cfg_teacher.keys():
            print("%s configuration:" % key, cfg_teacher[key])
            print('\n')

    assert cfg is not cfg_teacher
    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    """turn on the distillation mode"""
    cfg.TRAIN.DISTILL = True
    cfg_teacher.TRAIN.DISTILL = True
    net = build_network(settings.script_name, cfg)
    net_teacher = build_network(settings.script_teacher, cfg_teacher, teacher=True)

    # wrap networks to distributed one
    net.cuda()
    net_teacher.cuda()
    net_teacher.eval()

    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS)
        net_teacher = DDP(net_teacher, device_ids=[settings.local_rank])
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    # Loss functions and Actors
    if settings.script_name == 'mixformer2_vit':
        objective = {'ciou': ciou_loss, 'l1': l1_loss}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 
                       'corner': cfg.TRAIN.CORNER_WEIGHT}
        actor = MixFormerDistillStage1Actor(net=net, net_teacher=net_teacher, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=False,
                                      z_size_teacher=cfg_teacher.DATA.TEMPLATE.SIZE, x_size_teacher=cfg_teacher.DATA.SEARCH.SIZE, feat_sz=cfg.MODEL.FEAT_SZ)
    elif settings.script_name == 'mixformer2_vit_stu':
        objective = {'ciou': ciou_loss, 'l1': l1_loss}
        loss_weight = {'ciou': cfg.TRAIN.IOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 
                       'corner': cfg.TRAIN.CORNER_WEIGHT, 'feat': cfg.TRAIN.FEAT_WEIGHT}
        actor = MixFormerDistillStage2Actor(net=net, net_teacher=net_teacher, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=False,
                                            distill_layers_student=cfg.TRAIN.DISTILL_LAYERS_STUDENT, distill_layers_teacher=cfg.TRAIN.DISTILL_LAYERS_TEACHER)
    else:
        raise ValueError("illegal script name")
    if is_main_process():
        print("Loss weight: ", loss_weight)
        print("Actor: ", actor)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)

    remove_mode = (len(cfg.TRAIN.REMOVE_LAYERS) > 0)
    print("Remove mode: ", remove_mode)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler=lr_scheduler, use_amp=use_amp, remove_mode=remove_mode)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, distill=True)
