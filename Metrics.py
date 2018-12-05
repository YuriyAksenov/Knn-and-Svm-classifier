from typing import List


class Metrics:

    @staticmethod
    def p_value(real_data: list, predicted_data: list, labels: list):
        degrees_of_freedom, x_square = Metrics.__get_x_square_distribution(
            real_data, predicted_data, labels)
        return Metrics.__p_value(degrees_of_freedom, x_square)

    @staticmethod
    def f_score(real_data: list, predicted_data: list):
        """Get f score"""
        metrics = Metrics.__get_metrics(real_data, predicted_data)

        true_positive = metrics[0]
        false_positive = metrics[1]
        #true_negative = metrics[2]
        false_negative = metrics[3]

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fscore = 2*(precision*recall) / (precision+recall)

        return (fscore)

    @staticmethod
    def plot_confusion_matrix(real_data: list, predicted_data: list):
        import numpy as np
        metrics = Metrics.__get_metrics(real_data, predicted_data)
        aplot_confusion_matrix(cm=np.array([[metrics[0], metrics[1]],
                                            [metrics[3], metrics[2]]]), normalize=True, target_names=["0", "1"], title="Confusion Matrix, Normalized")

    @staticmethod
    def __get_x_square_distribution(real_data: list, predicted_data: list, labels: list):
        """Get  x^2 = Σ((o-e)2/e)"""
        degrees_of_freedom = len(labels) - 1
        result = 0.0
        for label in labels:
            # количество данных, принадлежащих данному классу реальных
            real_count = len([x for x in real_data if x == label])
            # количество данных, принадлежащих данному классу предсказанных
            predicted_count = len([x for x in predicted_data if x == label])
            result += (predicted_count - real_count)**2 / real_count
        return (degrees_of_freedom, result)

    @staticmethod
    def __p_value(degrees_of_freedom: int, x_square_distribution: float):
        table = Metrics.__p_value_table()
        for index, x in enumerate(table[degrees_of_freedom]):
            if x_square_distribution > x:
                continue
            if(index > 0):
                return (table[0][index], table[0][index-1])
            else:
                return (table[0][index], table[0][index])

    @staticmethod
    def __get_metrics(real_data: list, predicted_data: list):
        """Get metrics (true_positive, false_positive, true_negative, false_negative)"""
        small_number = 0.000000000000000001
        true_positive = small_number
        false_positive = small_number
        true_negative = small_number
        false_negative = small_number

        for i in range(len(predicted_data)):
            if (real_data[i] == 0) and (predicted_data[i] == 0):
                true_negative += 1
            if (real_data[i] == 0) and (predicted_data[i] == 1):
                false_positive += 1
            if (real_data[i] == 1) and (predicted_data[i] == 1):
                true_positive += 1
            if (real_data[i] == 1) and (predicted_data[i] == 0):
                false_negative += 1
        return (true_positive, false_positive, true_negative, false_negative)

    @staticmethod
    def __p_value_table():
        return [[0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01],
            [0.000, 0.001, 0.004, 0.016, 2.706, 3.841, 5.024, 6.635]]


def aplot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


# import numpy as np

# r1 = np.zeros((1, 100), dtype=np.int32).ravel().tolist()
# r2 = np.ones((1, 50), dtype=np.int32).ravel().tolist()
# a = r1+r2
# p1 = np.zeros((1, 90), dtype=np.int32).ravel().tolist()
# p2 = np.ones((1, 60), dtype=np.int32).ravel().tolist()
# b = p1+p2

# # x_square = Metrics.get_x_square_distribution(a, b, list([0, 1]))
# # print(x_square)
# # p = Metrics.p_value(x_square[0], x_square[1])
# # print(p)
# res = Metrics.p_value(a, b,  list([0, 1]))
# print(res)
# res = Metrics.f_score(a, b)
# print(res)
# res = Metrics.plot_confusion_matrix(a, b)
# print(res)
