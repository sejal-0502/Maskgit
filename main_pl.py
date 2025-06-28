import argparse, os, sys, datetime, glob, importlib, subprocess
import datetime
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.profilers import PyTorchProfiler, AdvancedProfiler
from callbacks.logging import ImageLogger
import sys
sys.path.append('../')
import visual_tokenization
from visual_tokenization.callbacks import SetupCallback, PeriodicCheckpoint, SaveCheckpointEveryNSteps
from visual_tokenization.taming.data.utils import custom_collate
sys.path.append('./')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='An output with one or more elements')


def get_jobid():
    try:
        jobid = os.environ["PBS_JOBID"].replace(".lmbtorque.informatik.uni-freiburg.de", "")
    except KeyError:
        try:
            if os.environ["SLURM_CLUSTER_NAME"].startswith("kislurm"):
                jobid = "DLC" + os.environ["SLURM_JOB_ID"]
            elif os.environ["SLURM_CLUSTER_NAME"].startswith("uc"):
                jobid = "BWUC" + os.environ["SLURM_JOB_ID"]
            elif os.environ["SLURM_CLUSTER_NAME"].startswith("jw") or os.environ["SLURM_CLUSTER_NAME"].startswith("juwels"):
                jobid = "JW" + os.environ["SLURM_JOB_ID"]
            else:
                jobid = "UNK" + os.environ["SLURM_JOB_ID"]
            # try:
            #     jobid = "DLC" + os.environ["SLURM_JOB_ID"]
        except KeyError:
            try:
                jobid = os.uname()[1]
            except KeyError:
                jobid = "local"
    return jobid


def get_git_last_commit():
    try:
        return subprocess.check_output(["git", "log", "--format=%H %s", "-n", "1"]).strip().decode("utf-8")
    except FileNotFoundError:
        return "no git"

def get_git_has_uncommitted_changes():
    # Run the git diff command with the --exit-code option
    try:
        result = subprocess.run(["git", "diff", "--exit-code", "HEAD"], stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        return False
    # Check the return code (0 means no differences, 1 means differences)
    return result.returncode != 0


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-g",
        "--n_gpus",
        type=int,
        default=None,
        help="number of gpus per node",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=None,
        help="number of nodes",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="check validation every n epochs",
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        default=30,
        help="number of epochs",
    )
    parser.add_argument(
        '--limit_train_batches',
        type=int,
        default=None,
        help='limit the number of training batches (see lightning Trainer docs)',
    )
    parser.add_argument(
        '--limit_val_batches',
        type=int,
        default=None,
        help='limit the number of validation batches (see lightning Trainer docs)',
    )
    parser.add_argument(
        "--no_test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        # "-d",
        "--dbg",
        action="store_true",
        help="custom debug mode...",
    )
    parser.add_argument(
        # lightning trainer arg for quick debugging, runs 1 batch and validates, no checkpointing, etc.
        "--fast_dev_run",
        action="store_true",
    )
    parser.add_argument(
        "--profiler",
        type=str2bool,
        nargs="?",
        default=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=2000,
        help="log image frequency",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))



class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size=None, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, dbg=False):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.dbg = dbg

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        if self.dbg:
            sampler = DistributedSampler(self.datasets["train"], shuffle=True) # if self.trainer.use_ddp else None
            return DataLoader(self.datasets["train"], batch_size=self.batch_size, 
                              num_workers=self.num_workers, shuffle=False, collate_fn=custom_collate, pin_memory=True, drop_last=True, sampler=sampler)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate, pin_memory=True, drop_last=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, pin_memory=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.val_batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    

    opt, unknown = parser.parse_known_args()

    if opt.n_gpus is None:
        # use slurm env vars if available otherwise use all CUDA_AVAILABLE gpus
        opt.n_gpus = int(os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
    if opt.n_nodes is None:
        # use slurm env vars if available otherwise set to 1
        opt.n_nodes = int(os.environ.get("SLURM_NNODES", 1))

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            # determine logdir from checkpoint
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
            print(f"Resuming from checkpoint {opt.resume}, logdir: {logdir}")
        else:
            # determine checkpoint from logdir
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
            print(f"Resuming from folder {opt.resume}, logdir: {logdir}")
        opt.resume_from_checkpoint = ckpt

        # determine configs from logdir
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base

        # determine name from logdir
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix+"_"+get_jobid()
        try:
            logdir = os.path.join(os.environ.get("MASKGIT_WORK_DIR", "logs"), nowname)
        except KeyError:
            logdir = os.path.join("logs", nowname)
        
    print(">>> Logging to {}".format(logdir))
    print(">>> Last git commit: {} {}".format(get_git_last_commit(), "*with uncommitted changes*" if get_git_has_uncommitted_changes() else ""))

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed, workers=True)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        
        

        # for k in nondefault_trainer_args(opt):
        #     trainer_config[k] = getattr(opt, k)
        # trainer_config["gpu"] = 0
        if opt.n_gpus == 0:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            # gpuinfo = trainer_config["gpus"]
            # print(f"Running on GPUs {gpuinfo}")
            cpu = False
        # trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # logger
        trainer_kwargs["logger"] = TensorBoardLogger(logdir, name="tb")
        print("Logging to {}".format(logdir))
        print("Logging with name {}".format("tb"))
        trainer_kwargs["strategy"] =  DDPStrategy(find_unused_parameters=True) #DDPStrategy()   #DDPStrategy(find_unused_parameters=True, static_graph=True) #"ddp"  #'ddp_find_unused_parameters_true'  DeepSpeedStrategy() # find_unused_parameters=True, static_graph=True
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = opt.n_gpus
        trainer_kwargs["num_nodes"] = opt.n_nodes
        trainer_kwargs["check_val_every_n_epoch"] = opt.check_val_every_n_epoch
        trainer_kwargs["max_epochs"] = opt.num_epochs
        trainer_kwargs["fast_dev_run"] = opt.fast_dev_run
        trainer_kwargs["limit_train_batches"] = opt.limit_train_batches
        trainer_kwargs["limit_val_batches"] = opt.limit_val_batches
        trainer_kwargs["use_distributed_sampler"] = not opt.dbg
        trainer_kwargs["precision"] = config.model.get("precision", "16-mixed")
        trainer_kwargs["gradient_clip_val"] = config.model.get("grad_clip", 1.0)
        trainer_kwargs["accumulate_grad_batches"] = config.model.get("grad_acc_steps", 1)

        # profiler        
        if opt.profiler:
            # profiler = PyTorchProfiler(
            # on_trace_ready = torch.profiler.tensorboard_trace_handler(logdir),
            # schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=3, repeat=2),
            # # record_functions=
            # )
            # trainer_kwargs["profiler"] = profiler
            # trainer_kwargs["profiler"] = "simple"
            from callbacks.profilers import SimpleProfilerCustom
            profiler = SimpleProfilerCustom(dirpath=logdir, filename="profile", burn_in_dict={"run_training_batch": 100})
            trainer_kwargs["profiler"] = profiler

        # trainer_kwargs["plugins"]= [SLURMEnvironment(requeue_signal=signal.SIGHUP)]
        
        # adding callbacks
        trainer_kwargs["callbacks"] = [
            SetupCallback(resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config),
            # SetupCallback(now=now, logdir=logdir, ckptdir=ckptdir, cfgdir=cfgdir, config=config, lightning_config=lightning_config),
            ImageLogger(batch_frequency=opt.log_frequency, max_images=10, clamp=True, increase_log_steps=False),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=ckptdir, filename='checkpoint-{epoch}-{step}', every_n_train_steps=2000, save_top_k=3, save_last=True, enable_version_counter=False, monitor="train/ce_loss"),
            ModelCheckpoint(dirpath=ckptdir, filename='checkpoint-{epoch}', every_n_epochs=1, save_top_k=3, save_last=True, enable_version_counter=False, monitor="val/ce_loss"),
            TQDMProgressBar(refresh_rate=100 if os.environ.get("SLURM_JOB_ID") else 1),
            PeriodicCheckpoint(every=5, end=150, dirpath=ckptdir)
        ]
        # if opt.tsne_epoch_frequency is not None and opt.tsne_epoch_frequency > 0:
        #     trainer_kwargs["callbacks"].append(CodebookTSNELogger(epoch_frequency=opt.tsne_epoch_frequency))

        trainer = Trainer( **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        print(f"{datetime.datetime.now()} - DATA CLASS INSTANTIATED")
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        # data.prepare_data()
        # print(f"{datetime.datetime.now()} - DATA PREPARED")
        data.setup()
        print(f"{datetime.datetime.now()} - DATA SETUP COMPLETE")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        model.configure_learning_rate(base_lr, bs, opt.n_gpus if not cpu else 1)
        #model.configure_lr_scheduler(config.model.get("warmup_steps", 0), config.model.get("min_lr_multiplier", 1))
        model.num_iters_per_epoch = len(data.datasets["train"]) // (config.data.params.batch_size * opt.n_gpus * config.model.get("grad_acc_steps", 1))
        print(f"Num iters per epoch: {model.num_iters_per_epoch}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                if opt.resume:
                    # model = model.load_from_checkpoint(opt.resume, strict=False)
                    # trainer.fit(model, data)
                    trainer.fit(model, data, ckpt_path = opt.resume_from_checkpoint)
                else:
                    trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
    
