from typing import List


class Metrics:

    @staticmethod
    def t_test(real_data: list, predicted_data: list):
        """Wilcoxon t-test for binary labels, so for diversity -1 0 1"""
        participators_count, t_empirical = Metrics.__t_test_empirical(real_data, predicted_data)
        table_row = Metrics.__t_value_table()[participators_count]
        t_critical_005 = table_row[1]
        t_critical_001 = table_row[2]
        if(t_empirical < t_critical_005 or t_critical_001 < t_empirical ):
            return "improved"
        else:
            return "not_improved"

    @staticmethod
    def __t_test_empirical(real_data: list, predicted_data: list):
        """Wilcoxon t-test for binary labels, so for diversity -1 0 1"""
        from collections import Counter

        absolute_diversity = []

        for i in range(len(real_data)):
            div_abs = abs(predicted_data[i] - real_data[i])
            if(div_abs != 0):
                absolute_diversity.append(div_abs)
        participators_count = len(absolute_diversity)
        if(participators_count > 50):
            print("Error there is no table more than 50 elements")
            return 0.0;
        t_empirical = sum(absolute_diversity)
        return (participators_count, t_empirical)

    @staticmethod
    def p_value(real_data: list, predicted_data: list, labels_count: int):
        degrees_of_freedom, x_square = Metrics.__get_x_square_distribution(
            real_data, predicted_data, labels_count)
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
    def __get_x_square_distribution(real_data: list, predicted_data: list, labels_count: int):
        """Get  x^2 = Σ((o-e)2/e)"""
        degrees_of_freedom = labels_count - 1
        result = 0.0
        for label in range(labels_count):
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
        """
        При необходимости дополнить таблицу)
        Таблица получения p value по хи квадрат. Первая строка  - заголовок, точнее p. 
        Вторая и далее строки - степени свободы. Первая строк - первая степень. 
        Значения в строке - это хи квадрат
        """
        return [[0.99, 0.975, 0.95, 0.90, 0.10, 0.05, 0.025, 0.01],
                [0.000, 0.001, 0.004, 0.016, 2.706, 3.841, 5.024, 6.635]]

    @staticmethod
    def __t_value_table():
        """
        Т-критерий Уилкоксона таблица
        | число исследуемых n  | p = 0.05 | p = 0.1 |
        """
        return [
            [5, 0, 0],
            [6, 2, 0],
            [7, 3, 0],
            [8, 5, 1],
            [9,  8,	3],
            [10, 10, 5],
            [11, 13, 7],
            [12, 17, 9],
            [13, 21, 12],
            [14, 25, 15],
            [15, 30, 19],
            [16, 35, 23],
            [17, 41, 27],
            [18, 47, 32],
            [19, 53, 37],
            [20, 60, 43],
            [21, 67, 49],
            [22, 75, 55],
            [23, 83, 62],
            [24, 91, 69],
            [25,100, 76],
            [26,110, 84],
            [27,119, 92],
            [28,130, 101],
            [29,140, 110],
            [30,151, 120],
            [31,163, 130],
            [32,175, 140],
            [33,187, 151],
            [34,200, 162],
            [35,213, 173],
            [36,227, 185],
            [37,241, 198],
            [38,256, 211],
            [39,271, 224],
            [40,286, 238],
            [41,302, 252],
            [42,319, 266],
            [43,336, 281],
            [44,353, 296],
            [45,371, 312],
            [46,389, 328],
            [47,407, 345],
            [48,426, 362],
            [49,446, 379],
            [50,466, 397]
        ]


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
# b = r1+r2 #p1+p2

# # x_square = Metrics.get_x_square_distribution(a, b, list([0, 1]))
# # print(x_square)
# # p = Metrics.p_value(x_square[0], x_square[1])
# # print(p)



# res = Metrics.p_value(a, b,  2)
# print(res)
# # res = Metrics.f_score(a, b)
# # print(res)
# # res = Metrics.plot_confusion_matrix(a, b)
# # print(res)

# res = Metrics.t_test(a,b)
# print(res)

# print(-1 - (-1))
