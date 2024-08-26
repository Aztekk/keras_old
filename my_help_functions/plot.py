import matplotlib.pyplot as plt

def plot_loss(history):
    '''
        history - объект, возвращаемый методом fit
    '''
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Потери на этапе обучения")
    plt.plot(epochs, val_loss_values, "b", label="Потери на этапе проверки")
    plt.title("Потери на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.xticks(range(0, len(history.history['loss']) + 2, 2))
    plt.ylabel("Потери")
    plt.legend()
    plt.show()
    
def plot_metric(history, metric, label):
    '''
        history - объект, возвращаемый методом fit
    '''
    history_dict = history.history
    train_metric = history_dict[metric]
    val_metric = history_dict[f"val_{metric}"]
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, "bo", label=f"{label} на этапе обучения")
    plt.plot(epochs, val_metric, "b", label=f"{label} на этапе проверки")
    plt.title(f"{label} на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.xticks(range(0, len(history.history['loss']) + 2, 2))
    plt.ylabel(label)
    plt.legend()
    plt.show()