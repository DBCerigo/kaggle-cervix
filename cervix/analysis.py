import matplotlib.pyplot as plt

def plot_log_errors(history):
    plt.plot(history.history['loss'])
    if history.history['val_loss']:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
