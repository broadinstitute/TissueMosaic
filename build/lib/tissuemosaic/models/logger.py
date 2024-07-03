import os
import shutil
import re
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from neptune.types import File


def copy_to_location_if_dir_exists(src_path, destination_path):
    """ Only if in cromwell, copies the most recent ckpt to path_to_cromwell_checkpointing """
    ckpt_dir = os.path.dirname(destination_path)
    if os.path.isdir(ckpt_dir):
        shutil.copyfile(src_path, destination_path)
        print("copied ckpt file from {0} to {1}".format(src_path, destination_path))
    else:
        print("NOT copied ckpt file {0} to {1} b/c destination don't exist".format(src_path, destination_path))


class NeptuneLoggerCkpt(NeptuneLogger):
    """ Thin wrapper around the Neptune Logger with the after_save_checkpoint specified """

    verbose = False

    # Use these two class variables to copy only the most recent ckpt to be used to resume simulation if pre-emption
    path_to_cromwell_checkpointing = '/cromwell_root/my_checkpoint.ckpt'
    mtime_cromwell_checkpointing = -100.0

    def __init__(self, **kargs):
        super(NeptuneLoggerCkpt, self).__init__(**kargs)
        self.last_model_mtime = None
        self.best_model_mtime = None
        self.best_k_models = dict()  # dictionary with path (key) and score (value)

    def delete_file_in_neptune(self, checkpoints_namespace, file_just_added):
        """
        Compare file already uploaded in neptune with the one recently added.
        If some diff file have the same beginning as a file recently added I remove the old files.
        """

        # the file to potentially drop are the difference between everything in Neptune and the one just added
        if self.experiment.exists(checkpoints_namespace):
            exp_structure = self.experiment.get_structure()
            uploaded_model_names = self._get_full_model_names_from_exp_structure(exp_structure, checkpoints_namespace)
        else:
            uploaded_model_names = []
        set_to_potentially_drop = set(uploaded_model_names) - set(file_just_added)

        # figure out the basename just added
        pattern_to_remove = r'-epoch=.*'
        base_name_just_added = [re.sub(pattern_to_remove, '', name) for name in file_just_added]

        # decide if to drop a file
        for file_to_drop in set_to_potentially_drop:
            for base_name in base_name_just_added:
                if file_to_drop.startswith(base_name):
                    del self.experiment[f"{checkpoints_namespace}/{file_to_drop}"]
                    if self.verbose:
                        print("deleted file", file_to_drop)

    def log_long_text(self, text, log_name):
        for x in text.split('\n'):
            # replace leading spaces with '-' character
            n = len(x) - len(x.lstrip(' '))
            token = '-' * n + x
            self.run[log_name].log(token)

    def log_model_summary(self, model, log_name: str = 'training/model/summary', max_depth=-1):
        model_str = str(ModelSummary(model=model, max_depth=max_depth))
        self.run[log_name] = File.from_content(content=model_str, extension="txt")

    def after_save_checkpoint(self, checkpoint_callback: "ReferenceType[ModelCheckpoint]") -> None:
        """
        Log checkpointed model.
        Called after model checkpoint callback saves a new checkpoint.
        Must work even when multiple ModelCheckPoint callbacks are present

        Args:
            checkpoint_callback: the model checkpoint callback instance
        """

        if not self._log_model_checkpoints:
            return

        file_names, file_paths, file_mtimes = [], [], []
        checkpoints_namespace = self._construct_path_with_prefix("model/checkpoints")

        # log best k models
        for path in list(checkpoint_callback.best_k_models.keys()):
            if os.path.exists(path):
                if path not in list(self.best_k_models.keys()):
                    name = self._get_full_model_name(path, checkpoint_callback)
                    mtime = os.path.getmtime(path)
                    file_names.append(name)
                    file_paths.append(path)
                    file_mtimes.append(mtime)
                    self.run[f"{checkpoints_namespace}/{name}"].upload(path)
        self.best_k_models = checkpoint_callback.best_k_models

        # log the best model
        path = checkpoint_callback.best_model_path
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if mtime != self.best_model_mtime and path not in self.best_k_models.keys():
                name = self._get_full_model_name(path, checkpoint_callback)
                file_names.append(name)
                file_paths.append(path)
                file_mtimes.append(mtime)
                self.run[f"{checkpoints_namespace}/{name}"].upload(path)
                self.best_model_mtime = mtime

        # retrieve info about last model
        path = checkpoint_callback.last_model_path
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if mtime != self.last_model_mtime and path not in self.best_k_models.keys():
                name = self._get_full_model_name(path, checkpoint_callback)
                file_names.append(name)
                file_paths.append(path)
                file_mtimes.append(mtime)
                self.run[f"{checkpoints_namespace}/{name}"].upload(path)
                self.last_model_mtime = mtime

        # DEBUG
        if self.verbose:
            print("uploaded ->", file_names)
            print(self.last_model_mtime)
            print(self.best_model_mtime)
            print(self.best_k_models)

        # delete file from neptune
        self.delete_file_in_neptune(checkpoints_namespace=checkpoints_namespace, file_just_added=file_names)

        # Updated the cromwell checkpointing file with the most recent file
        if len(file_mtimes) >= 1:
            max_mtime = max(file_mtimes)
            if max_mtime > NeptuneLoggerCkpt.mtime_cromwell_checkpointing:
                most_recent_file = file_paths[file_mtimes.index(max_mtime)]
                copy_to_location_if_dir_exists(src_path=most_recent_file,
                                               destination_path=NeptuneLoggerCkpt.path_to_cromwell_checkpointing)
                NeptuneLoggerCkpt.mtime_cromwell_checkpointing = max_mtime
                if self.verbose:
                    print("updated cromwell_time", NeptuneLoggerCkpt.mtime_cromwell_checkpointing)

