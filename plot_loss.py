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

    try:
        dense_1_loss = log['dense_1_loss']
        activation_1_loss = log['activation_1_loss']
    except ValueError:
        dense_1_loss = np.zeros((train_loss.shape[0],1))
        activation_1_loss = np.zeros((train_loss.shape[0],1))
        print("There is no logged value for separate losses :(")

    try:
        beta = log['beta']
        alpha = log['alpha']
    except ValueError:
        beta = np.zeros((train_loss.shape[0],1))
        alpha = np.zeros((train_loss.shape[0],1))
        print("There is no logged values for beta, alpha :(")
    
    try:
        lr = log['learning_rate']
    except ValueError:
        lr = np.zeros((train_loss.shape[0],1))
        print("There is no logged value for learning_rate :(")
    
    timesteps = list(range(train_loss.shape[0]))
    
    # Plot losses
    plt.subplot(121)
    plt.plot(timesteps, train_loss, 'r-', timesteps, val_loss, 'b-', timesteps, dense_1_loss, 'k-', timesteps, activation_1_loss, 'g-')
    plt.legend(["Training loss", "Validation loss", "Regression loss", "classification loss"])
    plt.ylabel('Losses')
    plt.xlabel('Epochs')
    plt.subplot(122)
    plt.plot(timesteps, beta, 'r-', timesteps, alpha, 'b-', timesteps, lr*1000, 'g-')
    plt.legend(["model-weight: beta", "model_weight: alpha", "learning rate*1000"])
    plt.ylabel('hyper_parameters')
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
