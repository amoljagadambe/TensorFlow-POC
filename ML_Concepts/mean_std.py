import numpy
import math

speed = [86, 87, 88, 86, 87, 85, 86]

x = numpy.std(speed)

print("standered deaviation by numy lib: ",x)

total_value = len(speed)
# print(total_value)
addition = sum(speed)
# print("additiion >>>", addition)
mean_value = addition / total_value
# print(mean_value)  # output: 86.42857142857143

squre_sum = []
for i in speed:
    normal = i - mean_value
    squre_sum.append(normal**2)

# print(sumsition_value)
print("standered deaviation by numy calculation: ",math.sqrt(sum(squre_sum) / total_value))