#!/usr/bin/env python

# This script trains a Self-Supervised Model using a folder containing anndata.h5ad files

import argparse
import torch
import sys
from typing import List
from datetime import timedelta

import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler, AdvancedProfiler
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from tissuemosaic.data import AnndataFolderDM
from tissuemosaic.models import NeptuneLoggerCkpt
from tissuemosaic.models.ssl_models import (
    Barlow,
    Dino,
    Vae,
    Simclr,
)


def smart_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def initialization(
        config_dict: dict,
        ckpt_file: str,
        initialization_type: str,
        ) -> (pl.LightningModule, pl.Trainer, dict, str):
    """
    Initialize all that is necessary for the simulation

    Args:
        config_dict: dictionary with all the parameters of the simulation
        ckpt_file: path to a ckpt file (obtained from trainer.save_checkpoint or the CheckpointCallback)
        initialization_type: str, either 'resume', 'extend', 'scratch', 'pretraining', 'predict_only'

    Returns:
        pl_model: the pytorch_lightning model either a scratch or one loaded from a checkpoint
        pl_trainer:  the pytorch_lightning trainer
        new_args_dict: dict with the configurations. This is identical to :attr:'config_dict'
            if :attr:'initialization_type' == 'scratch'.
            In other cases it can be different from :attr:'config_dict' because some entries are overwritten.
        ckpt_file_for_trainer: None or a str pointing to a ckpt_file to be used to resume the training.
    """
    assert initialization_type in {'resume', 'extend', 'scratch', 'pretraining', 'predict_only'}

    if initialization_type in {'resume', 'extend', 'pretraining', 'predict_only'}:
        assert ckpt_file is not None
    elif initialization_type == 'scratch':
        ckpt_file = None

    if initialization_type in {'resume', 'extend'}:
        ckpt_file_for_trainer = ckpt_file
    else:
        ckpt_file_for_trainer = None

    # Redefine the model
    if config_dict["ssl_model"] == "barlow":
        Model = Barlow
    elif config_dict["ssl_model"] == "dino":
        Model = Dino
    elif config_dict["ssl_model"] == "vae":
        Model = Vae
    elif config_dict["ssl_model"] == "simclr":
        Model = Simclr
    else:
        raise Exception("Invalid model value. Received {0}".format(config_dict["model"]))

    if initialization_type in {'resume'}:
        # use ckpt_file for everything
        pl_model = Model.load_from_checkpoint(checkpoint_path=ckpt_file)
        # TODO: I am having trouble using the same run_id
        neptune_run_id = None  # pl_model.neptune_run_id
        new_dict = pl_model.__dict__['_hparams'].copy()

    elif initialization_type in {'extend'}:
        # use ckpt_file only but overwrite stuff relative to the duration of the training
        pl_model = Model.load_from_checkpoint(checkpoint_path=ckpt_file)
        # TODO: I am having trouble using the same run_id
        neptune_run_id = None  # pl_model.neptune_run_id
        new_dict = pl_model.__dict__['_hparams'].copy()
        for key, value in config_dict.items():
            if key in {"max_epochs", "max_time_minutes"}:
                new_dict[key] = value

    elif initialization_type in {'scratch'}:
        # use args only
        pl_model = Model(**config_dict)
        neptune_run_id = None
        new_dict = config_dict.copy()

    elif initialization_type in {'pretraining'}:
        # use checkpoint but overwrite
        pl_model = Model.load_from_checkpoint(checkpoint_path=ckpt_file, **config_dict)
        neptune_run_id = None
        new_dict = config_dict.copy()

    else:
        raise Exception("You should not be here initialization_type = {0}".format(initialization_type))

    print("new_dict ->", new_dict)

    pl_neptune_logger = NeptuneLoggerCkpt(
        project=new_dict['neptune_project'],
        run=neptune_run_id,  # pass something here to keep logging onto an existing run
        log_model_checkpoints=True,  # copy the checkpoints into Neptune
        # neptune kargs
        mode="async" if config_dict.get("remote_logging", False) else "offline",
        tags=[str(config_dict.get("ssl_model", "unknown_model"))],
        source_files=["*.py", "*.yaml"],
        fail_on_exception=True  # it does not good if you are not logging anything but simulation keeps going,
    )

    if new_dict["profiler"] == 'advanced':
        profiler = AdvancedProfiler(dirpath="./", filename="advanced_profiler.out", line_count_restriction=1.0)
    elif new_dict["profiler"] == "simple":
        profiler = SimpleProfiler(dirpath="./", filename="simple_profiler.out")
    else:
        profiler = PassThroughProfiler()

    
    if torch.cuda.device_count() == 0:
        # TODO:
        #   This is supposed to mimic a multi-gpu training on CPU for debugging purposes
        #   Unfortunately it hangs. Lightining removed the ddp_cpu. Investigate solution
        #   For now, I change parameter so that it runs a single-cpu training.
        #   Can still be used for debug but not the multi-gpu part.
        strategy = None  # 'ddp'
        accelerator = 'cpu'
        num_processes = 1  # 2
        sync_batchnorm = False
        precision = 32
    elif torch.cuda.device_count() == 1:
        # single gpu
        accelerator = None
        strategy = None
        num_processes = 1
        sync_batchnorm = False
        precision = new_dict["precision"]
    else:
        # TODO:
        # Validation hangs with multi-gpu training. Investigate solution
        # more thaN 1 gpu
        accelerator = None
        num_processes = 1
        sync_batchnorm = True
        precision = new_dict["precision"]
        if config_dict["ssl_model"] == "vae":
            # vae uses manual optimization. I need to set this flag to true
            strategy = DDPStrategy(find_unused_parameters=True) 
        else:
            # everything else works with automatic differentiation. Set this to false for speed
            # strategy = DDPStrategy(find_unused_parameters=False)
            strategy = "ddp" # defaults to false in lightning 2.0

    # monitor the learning rate. This will work both when manual or scheduler is used to change the learning rate.
    lr_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)

    # save the best ckpt according to the monitor quantity
    ckpt_save_best = ModelCheckpoint(
        filename="best_checkpoint-{epoch}",  # the extension .ckpt will be added automatically
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=False,
        monitor='train_loss',
        save_top_k=1,
        mode='min',
        every_n_epochs=100,  # this is how frequently I check the monitor quantity
    )

    # save ckpts every XXX minutes or YYY epochs (useful for resuming after pre-emption)
    ckpt_train_interval = ModelCheckpoint(
        filename="periodic_checkpoint-{epoch}",  # the extension .ckpt will be added automatically
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=False,
        # the following 2 are mutually exclusive. Determine how frequently to save
        # train_time_interval=timedelta(minutes=new_dict["checkpoint_interval_minutes"]),
        every_n_epochs=new_dict["checkpoint_interval_epochs"],
        
        save_top_k=-1 ## save all periodic checkpoints
    )

    # save ckpt at the end of training
    ckpt_train_end = ModelCheckpoint(
        save_weights_only=False,
        save_on_train_epoch_end=True,
        save_last=True,
        every_n_epochs=0,
    )
    ckpt_train_end.CHECKPOINT_NAME_LAST = 'my_checkpoint_last'  # the extension .ckpt will be added automatically
    
    ## TODO: debug validation with multi-gpu training
    pl_trainer = Trainer(
        default_root_dir="saved_ckpt",
        profiler=profiler,
        # num_nodes=num_processes,  # number of different machines, FOr us this is 1
        num_nodes=1, ##TODO: set this automatically to machine specs
        accelerator='gpu',#accelerator,
        gpus=new_dict["gpus"],
        check_val_every_n_epoch=new_dict["check_val_every_n_epoch"],
        callbacks=[ckpt_train_end, ckpt_train_interval, ckpt_save_best, lr_monitor],
        strategy = strategy,
        num_sanity_val_steps=0,
        # debugging
        # fast_dev_run=True,
        overfit_batches=new_dict["overfit_batches"],
        # Select how long to train for
        max_epochs=new_dict["max_epochs"],
        max_time=timedelta(minutes=new_dict["max_time_minutes"]),
        # other stuff
        logger=pl_neptune_logger,
        log_every_n_steps=100,
        num_processes=new_dict["gpus"], ## needs to be same as number of gpus otherwise client socket time out error
        sync_batchnorm=sync_batchnorm,
        precision=precision,  # if using P100 GPU reduce to 16 (half-precision) to have massive speedup
        # If model uses automatic_optimization -> you can use gradient clipping in the trainer.
        # Otherwise gradient clipping need to be performed internally in the model
        gradient_clip_val=new_dict["gradient_clip_val"] if pl_model.automatic_optimization else None,
        gradient_clip_algorithm=new_dict["gradient_clip_algorithm"] if pl_model.automatic_optimization else None,
        # these are for debug and make the model slower
        detect_anomaly=new_dict["detect_anomaly"],
        deterministic=new_dict["deterministic"],
        # track_grad_norm='inf',
    )

    return pl_model, pl_trainer, new_dict, ckpt_file_for_trainer


def run_simulation(config_dict: dict, datamodule: AnndataFolderDM):
    """
    Args:
        config_dict: dictionary with all the config parameters
        datamodule: a AnndataFolderDM datamodule for Self Supervised Learning
    """

    pl.seed_everything(seed=config_dict['random_seed'], workers=True)

    # Initialization need to handle 4 situations: "resume", "extend", "pretraining", "scratch"
    try:
        print("trying to restart from pre-emption checkpoint")
        model, trainer, hparam_dict, ckpt_file_trainer = initialization(
            ckpt_file="./preemption_ckpt.pt",
            config_dict=config_dict,
            initialization_type='resume',
        )
    except (Exception, LookupError) as e2:
        print(e2)
        # this can be: resume, pretraining or scratch
        print("trying to initialize from:", config_dict["initialization_type"])
        model, trainer, hparam_dict, ckpt_file_trainer = initialization(
            ckpt_file=None if config_dict["initialization_type"] == 'scratch' else "./old_run_ckpt.pt",
            config_dict=config_dict,
            initialization_type=config_dict["initialization_type"],
        )
    print("initialization done. I have a model, trainer, hparam_dict")

    if model.global_rank == 0:
        # log model and hyperparameters
        trainer.logger.log_hyperparams(hparam_dict)
        trainer.logger.log_model_summary(model=model, log_name="training/model/summary")
        trainer.logger.log_long_text(
            text=model.__str__(),
            log_name="training/model/architecture",
        )

        # log the transform into neptune
        trainer.logger.log_long_text(
            text=repr(datamodule.trsfm_train_local),
            log_name="transform/train_local",
        )
        trainer.logger.log_long_text(
            text=repr(datamodule.trsfm_train_global),
            log_name="transform/train_global",
        )
        trainer.logger.log_long_text(
            text=repr(datamodule.trsfm_test),
            log_name="transform/test",
        )

    # training begins
    if ckpt_file_trainer is not None:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_file_trainer)
    else:
        trainer.fit(model=model, datamodule=datamodule)

    # save the checkpint at the end of training
    # This is redundant if the Trainer has the ModelCHeckpoint callbacks
    trainer.save_checkpoint(config_dict.get("ckpt_out", "ckpt_last.pt"))

    # at the end close connection to neptune database
    if model.global_rank == 0:
        trainer.logger.finalize(status='completed')


def parse_args(argv: List[str]) -> dict:
    """
    Read argv from command-line and produce a configuration dictionary.
    If the command-line arguments include

    If the command-line arguments include '--to_yaml my_yaml_file.yaml' the configuration dictionary is written to file.

    Args:
        argv: the parameter passed from the command line.

    Note:
        If argv includes '--config input.yaml' the parameters are read from file.
        The config.yaml parameters have priority over the CLI parameters.

    Note:
        If argv includes '--to_yaml output.yaml' the configuration dictionary is written to file.

    Note:
        Parameters which are missing from both argv and config.yaml will be set to their default values.

    Returns:
        config_dict: a dictionary with all the configuration parameters.
    """
    parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
    parser.add_argument("--config", type=str, default=None,
                        help="If provided read the configuration file *.yaml. The parameters passed from config file \
                              have precedence over the one from the CLI")
    parser.add_argument("--to_yaml", type=str, default=None,
                        help="If provided write a yaml file with the parameters and exit.")
    parser.add_argument("--ckpt_out", type=str, default="ckpt_last.pt",
                        help="The path to save the checkpoint at the end of training.")

    # parameters for Neptune and random seed
    parser.add_argument("--random_seed", type=int, default=1, help="Integer specifying the global random seed")
    parser.add_argument("--neptune_project", type=str, default="skambham/tp-test",#"cellarium/tissue-mosaic",
                        help="neptune project name. This is where the data will be logged")

    # parameters for the trainer
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count(),
                        help="Number of gpus to use for training.")
    parser.add_argument("--max_epochs", type=int, default=3000, help="maximum number of training epochs")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=25, help="run validation every N epochs")
    parser.add_argument("--profiler", default="passthrough", type=str, choices=["passthrough", "simple", "advanced"])
    parser.add_argument("--max_time_minutes", type=int, default=1440,
                        help="Training will be stopped after this amount of time. \
                        It is safety mechanism to make sure that runs will not be going forever.")
    parser.add_argument("--checkpoint_interval_minutes", type=int, default=25,
                        help="checkpoint the simulation every N minutes")
    parser.add_argument("--precision", type=int, default=16, choices=[32, 16],
                        help="full or half precision. On CPU only machines must be 32")
    parser.add_argument("--overfit_batches", type=int, default=0,
                        help="Specify the number of batches to use to overfit the model. \
                        Usefull for debugging. If 0 no overfit.")
    parser.add_argument("--gradient_clip_val", type=float, default=0.5,
                        help="Clip the gradients to this value. If 0 no clipping")
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value", choices=["norm", "value"],
                        help="Algorithm to use for gradient clipping.")
    parser.add_argument("--detect_anomaly", type=smart_bool, default=False, help="Detect anomaly, i.e. Nans")
    parser.add_argument("--deterministic", type=smart_bool, default=False, help="Deterministic operation in CUDA?")

    # select the model and dataset
    parser.add_argument("--ssl_model", default="barlow", type=str, choices=["barlow", "dino", "vae", "simclr"],
                        help="Self Supervised Learning framework")

    # simulation parameters
    parser.add_argument("--initialization_type", type=str, default="scratch",
                        choices=["resume", "extend", "predict_only", "pretraining", "scratch"])
    parser.add_argument("--remote_logging", default=True, type=smart_bool,
                        help="If True the logs are saved on the Neptune server. Set to False to save logs locally")

    # Figure out which model I am using
    (args, _) = parser.parse_known_args(argv)  # args is a Namespace object

    if args.config is not None:
        import yaml
        with open(args.config, 'r') as stream:
            yaml_config_dict = yaml.safe_load(stream)
        delattr(args, 'config')
    else:
        yaml_config_dict = dict()

    # Decide which ssl_model to use
    ssl_model_from_yaml = yaml_config_dict.get("ssl_model", None)
    ssl_model = args.ssl_model if ssl_model_from_yaml is None else ssl_model_from_yaml

    # Update the parser with model params
    if ssl_model == "barlow":
        parser = Barlow.add_specific_args(parser)
    elif ssl_model == "dino":
        parser = Dino.add_specific_args(parser)
    elif ssl_model == 'vae':
        parser = Vae.add_specific_args(parser)
    elif ssl_model == 'simclr':
        parser = Simclr.add_specific_args(parser)
    else:
        raise Exception("Invalid ssl_model {0}".format(ssl_model))

    # Update parser with datamodule params
    parser = AnndataFolderDM.add_specific_args(parser)

    # Add help at the very end
    parser = argparse.ArgumentParser(parents=[parser], add_help=True)

    # Process everything.
    # Note that the priority order is:
    # config.yaml > CLI params > defaults value
    args = parser.parse_args(argv)
    config_dict_full = vars(args)  # this contains CLI params or their default
    config_dict_full.update(yaml_config_dict)  # override with the yaml config params
    return config_dict_full


if __name__ == '__main__':
    """ This is executed when run from the command line """
    config_dict_ = parse_args(sys.argv[1:])

    to_yaml_file = config_dict_["to_yaml"]
    if to_yaml_file is not None:
        print("Writing the config parameters to file ->", to_yaml_file)
        config_dict_.pop('to_yaml', None)
        config_dict_.pop('config', None)
        import yaml
        with open(to_yaml_file, 'w') as file:
            yaml.dump(config_dict_, file)
    else:
        # Run the simulation

        # Instantiate the datamodule
        datamodule_ = AnndataFolderDM(**config_dict_)

        # overwrite so that ch_in is the one specified by DM
        config_dict_["image_in_ch"] = datamodule_.ch_in

        # run the simulation
        run_simulation(config_dict=config_dict_, datamodule=datamodule_)
