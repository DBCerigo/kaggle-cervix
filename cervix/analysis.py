import pickle
import matplotlib.pyplot as plt

def plot_log_errors(history_dict):
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
    try:
        f = open(fp, 'wb')
        pickle.dump(history,f)
        f.close()
        return True
    except:
        return False
    
