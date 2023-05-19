# transpose와 reshape의 차이점
# transpose()는 데이터의 순서대로 바뀜
# reshape()는 데이터를 변행해서 바뀜
'''
import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print("Original : \n", a)
a_transpose = np.transpose(a)
print("Transpose : \n", a_transpose)
'''

import numpy as np
a = np.array([[1,2,3],[4,5,6]])
print("Original : \n", a)
a_reshape = np.reshape(a, (3,2))
print("Reshape : \n", a_reshape)