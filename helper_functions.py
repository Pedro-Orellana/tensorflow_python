import tensorflow as tf
import matplotlib.pyplot as plt

def plot_history_curves(history):
    """
    function to plot the loss and accuracy curves of a model.

    params:
    history: the model history object returned when fit() method is called on the model
    """

    #getting the loss curves
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    #getting accuracy curves
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    #getting epochs (x-axis)
    epochs = range(len(accuracy))

    #plotting loss curves
    plt.title("loss curves")
    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, val_loss, label="validation loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.figure()

    #plotting accuracy curves
    plt.title("accuracy curves")
    plt.plot(epochs, accuracy, label="training accuracy")
    plt.plot(epochs, val_accuracy, label="validation accuracy")
    plt.xlabel("epochs")
    plt.legend()
