"""
Learning rate scan
_____________________________________________________
Perform muliple trainings scanning through different 
learning rates to find optimum value
Best lr will have the minimum val loss
# TODO this should probably be test loss
"""

import numpy as np
from run.train import train
from scripts.utils import logger
import shutil
import os
import glob

def lr_scan(args):
    
    lr_range = np.linspace(args.lr_range[0],args.lr_range[1], args.lr_range[2])
    val_losses = []

    # Make directory to temporarily save weights to
    try:
        os.mkdir(os.path.join("network_weights", "tmp"))
    except FileExistsError:
        pass

    # Loop through learninig rates
    for lr in lr_range:
        args.lr = lr
        val_loss, _ = train(args)
        logger.log(f"Learning rate = {lr}  -- Val Loss = {val_loss}")
        val_losses.append(val_loss) 

        # If val_loss improved move weights from tmp to network_weights
        if val_loss <= min(val_losses):
            logger.log("Val loss is better than previous best - moving weights files")

            # Remove old weights
            old_weights = glob.glob(os.path.join("network_weights", "*.h5"))
            for file in old_weights:
                os.remove(file)
            
            # move new weights
            for file in glob.glob(os.path.join("network_weights", "tmp")):
                shutil.move(file, "network_weights")

    # Find best result and print
    min_loss_idx = np.argmin(val_losses)
    logger.log("\n\n ****************************")
    logger.log(f"Best learning rate = {lr_range[min_loss_idx]} -- Loss = {val_losses[min_loss_idx]}")