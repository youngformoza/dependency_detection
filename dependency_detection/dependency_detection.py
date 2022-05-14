import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import optimize
import csv
import os


def mnkGP(x_data, y_data, path):
    """
    метод наименьших квадратов
    :param x_data: значения x (независимая переменная)
    :param y_data: значения y (зависимая переменная)
    :param path: путь, по которому сохраняется изображение графика
    """
    d = 2
    fp, residuals, rank, sv, rcond = np.polyfit(x_data, y_data, d, full=True)
    f = np.poly1d(fp)
    print('Function: y = ' +
          str(round(fp[0], 4)) + ' * x^2 + (' + str(round(fp[1], 4)) + ') * x + (' + str(round(fp[2], 4)) + ')')
    y1 = [fp[0] * x_data[i] ** 2 + fp[1] * x_data[i] + fp[2] for i in range(0, len(x_data))]
    so = round(sum([abs(y_data[i] - y1[i]) for i in range(0, len(x_data))]) / (len(x_data) * sum(y_data)) * 100, 4)
    print('Average quadratic deviation ' + str(so))
    fx = np.linspace(x_data[0], x_data[-1] + 1, len(x_data))
    plt.plot(x_data, y_data, 'o', label='Original data', markersize=10)
    plt.plot(fx, f(fx), linewidth=2)
    plt.grid(True)
    plt.savefig(path + '_parabolic_model_2.png')
    plt.show()


def func_exp(x, a, b, c):
    """
    функция экспоненциальной зависимости
    :param x: значение x (независимая переменная)
    :param a: коэффициент a
    :param b: коэффициент b
    :param c: коэффициент c
    :return: значение функции при заданных параметрах
    """
    return a * np.exp(b * x) + c


def exponential_regression(x_data, y_data, path):
    """
    определение коэффициентов для построения экспоненциальной зависимости
    :param x_data: значения x (независимая переменная)
    :param y_data: значения y (зависимая переменная)
    :param path: путь, по которому сохраняется изображение графика
    """
    popt, pcov = optimize.curve_fit(func_exp, x_data, y_data, p0=(-1, 0.01, 1))
    print('Function: y = ' + str(round(popt[0], 4)) + '* e^(' + str(round(popt[1], 4)) + ' * x) + '
          '(' + str(round(popt[0], 4)) + ')')
    plt.plot(x_data, y_data, 'x', color='xkcd:maroon', label="data")
    plt.plot(x_data, func_exp(x_data, *popt), color='xkcd:teal', label="fit: {:.3f}, {:.3f}, {:.3f}".format(*popt))
    plt.legend()
    plt.savefig(path + '_exponential_model.png')
    plt.show()


def find_medians(df_dependencies, length_median):
    """
    поиск медианных значений на отсортированном по возрастанию списке
    :param df_dependencies: начальный датафрейм с зависимостями
    :param length_median: длина отрезка, на которых будет находиться медиана
    :return: список медианных значений
    """
    list_dependencies = df_dependencies.values.tolist()
    list_dependencies.sort(key=lambda x: x[0])

    list_medians = []
    count = 0

    if length_median % 2 == 0:
        for d in range(int(length_median / 2) - 1, len(list_dependencies), length_median):
            list_medians.append([])
            for i in range(2):
                list_medians[count].append(round((list_dependencies[d][i] + list_dependencies[d + 1][i]) / 2, 3))
            count += 1
    else:
        for d in range(int(length_median / 2) - 1, len(list_dependencies), length_median):
            list_medians.append([])
            for i in range(2):
                list_medians[count].append(list_dependencies[d][i])
            count += 1

    return list_medians


def main():
    columns = []

    print("Enter dependency using one of number of next parameters:\n1 - density\n2 - intensity\n3 - speed\n")

    choice_x = int(input("independent parameter (X): "))
    if choice_x == 1:
        columns.append("density")
    elif choice_x == 2:
        columns.append("intensity")
    elif choice_x == 3:
        columns.append("speed")
    else:
        print("Wrong number")

    choice_y = int(input("independent parameter (Y): "))
    if choice_x == choice_y:
        print("X can't be equal Y")
    else:
        if choice_y == 1:
            columns.append("density")
        elif choice_y == 2:
            columns.append("intensity")
        elif choice_y == 3:
            columns.append("speed")
        else:
            print("Wrong number")

    path_csv = input("Path to csv file to read: ")
    if os.access(path_csv, os.F_OK) is False:
        print("There isn't such file")
        return 0

    end_path = path_csv.rfind('\\')
    path = path_csv[0:(end_path + 1)]

    df_dependency = pd.read_csv(path_csv, names=columns, sep=';', decimal=',')
    df_dependency.drop((df_dependency.loc[df_dependency['density'] == 0]).index, inplace=True)

    length_median = int(input("length of segments to divide into: "))
    list_medians = find_medians(df_dependency, length_median)
    new_path = path + columns[0] + "_to_" + columns[1]
    new_filename = new_path + "_new.csv"

    with open(new_filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(columns)
        write.writerows(list_medians)

    choice_type = int(input("Parameters for which dependency do you want to know?\n0 - Parabolic\n1 - Exponential\n"))
    x_list = list(list_medians[i][0] for i in range(0, len(list_medians)))
    y_list = list(list_medians[i][1] for i in range(0, len(list_medians)))
    if choice_type:
        exponential_regression(np.array(x_list), np.array(y_list), new_path)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(20, 10))
        ax1.scatter(x_list, y_list)
        scatter = sns.regplot(x=x_list, y=y_list, order=2)
        fig.show()
        fig.savefig(new_path + '_parabolic_model_1.png')

        x_ar = np.array(x_list)
        y_ar = np.array(y_list)
        mnkGP(x_ar, y_ar, new_path)


if __name__ == '__main__':
    main()
