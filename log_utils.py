import logz
import numpy as np

import keras
from keras import backend as K



class MyCallback(keras.callbacks.Callback):
    """
    Customized callback class.
    
    # Arguments
       filepath: Path to save model.
       period: Frequency in epochs with which model is saved.
       batch_size: Number of images per batch.
    """
    
    def __init__(self, filepath, period, batch_size):
        self.filepath = filepath
        self.period = period
        self.batch_size = batch_size
        

    def on_epoch_begin(self, epoch, logs=None):
        
        # Decrease weight for binary cross-entropy loss
        beta_epoch = 1      # Epochs after which beta loss kicks in
        sess = K.get_session()
        self.model.beta.load(np.maximum(0.0, 1.0-np.exp(-1.0/beta_epoch*(epoch-beta_epoch))), sess)
        self.model.alpha.load(1.0, sess)

        print("alpha = ", self.model.alpha.eval(sess))
        print("beta = ", self.model.beta.eval(sess))
        self.scheduler(self.model, epoch, beta_epoch, new_lr=0.0001)        # Change new_lr 
        print("Learning Rate = ", K.get_value(self.model.optimizer.lr))


    def on_epoch_end(self, epoch, logs={}):
        
        # Save training and validation losses
        logz.log_tabular('train_loss', logs.get('loss'))
        logz.log_tabular('val_loss', logs.get('val_loss'))
        logz.dump_tabular()
        print("learning_rate = ", K.eval(self.model.optimizer.lr))
            
        # Save model every 'period' epochs
        if (epoch+1) % self.period == 0:
            filename = self.filepath + '/model_weights_' + str(epoch) + '.h5'
            print("Saved model at {}".format(filename))
            self.model.save_weights(filename, overwrite=True)

        # Hard mining
        sess = K.get_session()
        mse_function = self.batch_size-(self.batch_size-10)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
        entropy_function = self.batch_size-(self.batch_size-5)*(np.maximum(0.0,1.0-np.exp(-1.0/30.0*(epoch-30.0))))
        self.model.k_mse.load(int(np.round(mse_function)), sess)
        self.model.k_entropy.load(int(np.round(entropy_function)), sess)

    # Changing learning rate after beta loss is kicked in
    def scheduler(self, model, epoch, beta_epoch, new_lr):
        if epoch==beta_epoch:
            K.set_value(model.optimizer.lr, new_lr)
        
    