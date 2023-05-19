import numpy as np

oliers = np.array([-50, -10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, (25, 50, 75))
    print('1사분위 : ', quartile_1)
    print('2사분위 : ', q2)
    print('3사분위 : ', quartile_3)

    iqr = quartile_3 - quartile_1
    print('IQR : ', iqr)

    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    print('lower_bound : ', lower_bound)
    print('upper_bound : ', upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(oliers)
print('이상치의 위치 : ', outliers_loc)

# 1사분위 :  3.25
# 2사분위 :  6.5
# 3사분위 :  9.75
# IQR :  6.5
# lower_bound :  -6.5
# upper_bound :  19.5
# 이상치의 위치 :  (array([ 0,  1, 13], dtype=int64),)

#시각화
import matplotlib.pyplot as plt
plt.boxplot(oliers)
plt.show()