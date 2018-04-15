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

headers = ['Aorta_max', 'Pul_max', 'Aorta_mean', 'Pul_mean', 'Aorta_abs_mean', 'Pul_abs_mean', 'Aorta_std', 'Pul_std',
           'Aorta_var', 'Pul_var', 'Aorta_sum', 'Pul_sum']

# rows = [(12.45, 9, 0.72, 0.81, 1.57, 1.04, 2.48, 1.09, 75552.81, 36326.48, 0.72, 0.81),
# 		(18.14, 9.64, 0.75, 1.12, 1.61, 1.86, 2.6, 3.45, 87825.43, 55060.65, 0.75, 1.12),]

# with open('sur.csv', 'w', newline='') as f:
# 	f_csv = csv.writer(f)
# 	f_csv.writerow(headers)
# 	f_csv.writerows(rows)

data_read = []
data = []
with open('sur.csv') as f:
    f_csv = csv.reader(f)
    for i, row in enumerate(f_csv):
        data_read.append(row)

with open('test.csv', 'w', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(data_read)

# with open('sur.csv') as f:
# 	f_csv = csv.reader(f)
# 	for i, row in enumerate(f_csv):
# 		if i == 0:
# 			headers_read = row
# 		else:
# 			data.append(list(row))


print('finished')
