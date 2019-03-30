import csv

import matplotlib.pyplot as plt
import numpy as np


def converter(value):
    ee_position = value.find("E")
    e_position = value.find("e")
    e_pos = ee_position
    if e_position > ee_position:
        e_pos = e_position

    if e_pos > 0:
        _value = float(value[0:e_pos])
        return int(_value * 1000000)
    else:
        return int(value)


def simple_chart(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='GRADE', ylabel='PRICE USD',
           title='')
    ax.grid()

    # fig.savefig("results/test.png")
    plt.show()


def pie_chart(sizes):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = '1', '2', '3', '4', '5<'
    # sizes = [15, 30, 45, 10]
    explode = (0.05, 0.05, 0.05, 0.05, 0.05)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels,
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def bar_chart(x, y):
    print(x)
    print(y)
    plt.rcdefaults()
    fig, ax = plt.subplots()

    # Example data
    people = y
    y_pos = np.arange(len(people))
    performance = x

    ax.barh(y_pos, performance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(people)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('PRICE USD')
    ax.set_title('#ROOM = 2 , 1990 < YR_BUILT')

    plt.show()


def importer():
    x_array = []
    y_array = []
    w_array = []

    max_room = 10

    with open('data/kc_house_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[3] != "bedrooms" and int(row[3]) == 2:
                if 1990 < int(row[14]):
                    x_array.append(row[2])
                    y_array.append(row[16])

        # y_agg_array = [0 for x in range(0, max_room)]
        # y_avg_cnt_array = [0 for x in range(0, max_room)]
        #
        # for counter, value in enumerate(x_array):
        #     if counter > 1 and 4 < int(value) < 12:
        #         y_agg_array[int(value) - 5] += converter(y_array[counter])
        #         y_avg_cnt_array[int(value) - 5] += 1
        #
        # y_avg_array = [0 for x in range(0, max_room)]
        # for counter, value in enumerate(y_agg_array):
        #     y_avg_array[counter] = int(value / y_avg_cnt_array[counter])

        # simple_chart([x for x in range(5, 11)], y_avg_array[0:-1])

        # pie_array = [0 for x in range(0, max_room)]
        # for counter, value in enumerate(x_array):
        #     if counter > 1 and int(value) < max_room:
        #         if int(value) > 5:
        #             _value = 5
        #         else:
        #             _value = int(value) - 1
        #         pie_array[_value] += 1
        #
        # pie_chart(pie_array[0:5])
        # print(pie_array)

        len_zipcode = 70
        zipcode_array = []
        zipcode_cnt_array = [1 for x in range(0, len_zipcode)]
        zipcode_price_array = [0 for x in range(0, len_zipcode)]
        for counter, value in enumerate(y_array):
            if value not in zipcode_array:
                zipcode_array.append(value)
            else:
                zipcode_cnt_array[zipcode_array.index(value)] += 1
                zipcode_price_array[zipcode_array.index(value)] += converter(x_array[counter])

        zipcode_price_avg_array = []
        for counter, value in enumerate(zipcode_price_array):
            if counter < 7:
                zipcode_price_avg_array.append(int(value / zipcode_cnt_array[counter]))

    bar_chart(zipcode_price_avg_array, zipcode_array[0:7])


def main():
    importer()
    # print(converter("1.57e+06"))


if __name__ == "__main__":
    main()
