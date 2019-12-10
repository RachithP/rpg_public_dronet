import os
import sys
import numpy as np
import gflags
import matplotlib.pyplot as plt

from common_flags import FLAGS

        
def _main():
    
    # Read log file
    log_file = os.path.join(FLAGS.experiment_rootdir, "log.txt")
    try:
        log = np.genfromtxt(log_file, delimiter='\t',dtype=None, names=True)
    except:
        raise IOError("Log file not found")

    train_loss = log['train_loss']
    val_loss = log['val_loss']
    dense_1_loss = log['dense_1_loss']
    activation_1_loss = log['activation_1_loss']
    timesteps = list(range(train_loss.shape[0]))
    
    # Plot losses
    plt.plot(timesteps, train_loss, 'r-', timesteps, val_loss, 'b-', timesteps, dense_1_loss, 'k-', timesteps, activation_1_loss, 'g-')
    plt.legend(["Training loss", "Validation loss", "Regression loss", "classification loss"])
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.show()
    plt.savefig(os.path.join(FLAGS.experiment_rootdir, "log.png"))
    
        

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
