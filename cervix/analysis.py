import pickle
import matplotlib.pyplot as plt

def plot_log_errors(history):
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def save_history(history, fp):
    try:
        f = open(fp, 'wb')
        pickle.dump(history,f)
        f.close()
        return True
    except:
        return False
    
