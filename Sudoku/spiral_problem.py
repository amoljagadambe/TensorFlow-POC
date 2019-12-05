import numpy as np

spiral = np.arange(20).reshape(4,5)
print(spiral)

#print(spiral[3, 3])

row_start = 0
row_end = 4
col_start = 0
col_end = 5

while row_start < row_end and col_start < col_end:
    for i in range(col_start,col_end):
        print(spiral[col_start,i],end=" ")

    row_start +=1
    for i in range(row_start, row_end):
        print(spiral[i, col_end-1], end=" ")

    col_end -=1
    for i in range(col_end, col_start, -1):
        print(spiral[col_end-1, i-1], end=" ")

    row_end -=1
    for i in range(row_end,row_start,-1):
        print(spiral[i-1,row_start-1],end=" ")

    col_start +=1
