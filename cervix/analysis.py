import keras 
import pickle
import matplotlib.pyplot as plt

class History(keras.callbacks.Callback):
    def __init__(self):
        self.history = {'loss':[], 'val_loss':[]}

    def on_epoch_end(self, epoch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))


def plot_log_errors(history):
    history_dict = history.history
    # using dict as this is the object we will save(pickle)
    plt.plot(history_dict['loss'])
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def save_history(history, fp):
    history_dict = history.history
    try:
        f = open(fp, 'wb')
        pickle.dump(history_dict,f)
        f.close()
        return True
    except:
        return False
    
