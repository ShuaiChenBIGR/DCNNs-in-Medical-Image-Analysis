# =======================================================================================
# Aorta Segmentation Project                                                           #
#                                                                                      #
# 0. Test 1                                                                            #
#                                                                                      #
# created by                                                                           #
# Shuai Chen                                                                           #
# PhD student                                                                          #
# Medical Informatics                                                                  #
#                                                                                      #
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603   #
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands      #
# Email s.chen.2@erasmusmc.nl | Telephone +31 6 334 516 99                             #
# www.erasmusmc.nl                                                                     #
#                                                                                      #
# created on 26/10/2017                                                                #
# Last update: 26/10/2017                                                              #
# =======================================================================================
import csv
import numpy as np
import matplotlib.pyplot as plt


def box_plot_surface_distance(data, labels, row, col, fs):

    flierprops = dict(marker='o',  markersize=4,
                      linestyle='none')
    medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')

    fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(7.5, 5))

    axes[0, 0].boxplot(data[:, 0:2], labels=labels, showmeans=True, meanline=True, flierprops=flierprops, medianprops=medianprops)
    axes[0, 0].set_title('max', fontsize=fs)
    axes[0, 0].set_ylabel('voxels')

    axes[0, 1].boxplot(data[:, 2:4], labels=labels, showmeans=True, meanline=True, flierprops=flierprops, medianprops=medianprops)
    axes[0, 1].set_title('mean', fontsize=fs)
    axes[0, 1].set_ylabel('voxels')


    axes[0, 2].boxplot(data[:, 4:6], labels=labels, showmeans=True, meanline=True, flierprops=flierprops, medianprops=medianprops)
    axes[0, 2].set_title('abs_mean', fontsize=fs)
    axes[0, 2].set_ylabel('voxels')


    axes[1, 0].boxplot(data[:, 6:8], labels=labels, showmeans=True, meanline=True, flierprops=flierprops, medianprops=medianprops)
    axes[1, 0].set_title('std', fontsize=fs)
    axes[1, 0].set_ylabel('voxels')


    axes[1, 1].boxplot(data[:, 8:10], labels=labels, showmeans=True, meanline=True, flierprops=flierprops, medianprops=medianprops)
    axes[1, 1].set_title('var', fontsize=fs)
    axes[1, 1].set_ylabel('voxels')


    axes[1, 2].boxplot(data[:, 10:12], labels=labels, showmeans=True, meanline=True, flierprops=flierprops, medianprops=medianprops)
    axes[1, 2].set_title('sum', fontsize=fs)
    axes[1, 2].set_ylabel('voxels')
    axes[1, 2].set_yticklabels([])

    # for ax in axes.flatten():
        # ax.set_yscale('log')
        # ax.set_yticklabels([])

    fig.subplots_adjust(wspace=0.8, hspace=0.4)
    plt.show()


headers_read = []
data_list = []
with open('test.csv') as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        if i == 0:
            headers_read = row
        else:
            data_list.append(list(row))

data = np.asarray(data_list, dtype=np.float32)

labels = ['Aorta', 'Pul']

box_plot_surface_distance(data, labels, 2, 3, 10)

print('Box plot finished')
