import os, sys
import signal

# TODO: Use installed CPCargo package, after CPCargo PR #3 or #4 is merged
sys.path.append("../CPCargo/src") # use local run_cfg (from same directory)

from CPCargo import CheckpointCargo, Heartbeat

import pytorch_lightning
from threestudio.utils.misc import parse_version

if parse_version(pytorch_lightning.__version__) > parse_version("1.8"):
    from pytorch_lightning.callbacks import Callback
else:
    from pytorch_lightning.callbacks.base import Callback

from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn

SignalCheckpoint = False

class CPCargoCheckpointing:
    def __init__(self, dest_s3region, dest_s3bucket, ckpts_dir, timeout, callbacks):
        # Set up saving to S3 via CPCargo
        # Set a signal handler for sigterm to capture termination request
        signal.signal(signal.SIGTERM, signal_handler)
        os.makedirs(ckpts_dir, exist_ok = False) 
        checkpoint_path = ckpts_dir
        dest_s3_path = f"s3://{dest_s3bucket}/threestudio-ckpts"
        CP = CheckpointCargo(src_dir=checkpoint_path,
                            dst_url=dest_s3_path,
                            region=dest_s3region,
                            file_regex=r'.*',
                            recursive=True)

        # You can start monitoring checkpoint directory after data pipeline subprocesses forks
        CP.start()
        callbacks += [HeartbeatCallback(timeout)]

def signal_handler(signum, frame):
  # trigger a checkpoint and then exit cleanly
  # do kill -15 <pid> from a separate shell to trigger checkpoint-then-exit behavior
  #logger.info("Got signal {sig}".format(sig=signal.Signals(signum).name))
  if signum == signal.SIGTERM:
    global SignalCheckpoint
    SignalCheckpoint = True

class HeartbeatCallback(Callback):
    def __init__(self, timeout):
        super().__init__()
        self.hb = Heartbeat(timeout=timeout)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        self.hb.pulse()

