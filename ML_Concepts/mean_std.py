import numpy

speed = [86, 87, 88, 86, 87, 85, 86]

x = numpy.std(speed)

print("STANDARD DEVIATION by numpy library: ", x)


def meanCalualation(list_data=[]):
    """"
       Compute the mean

       Returns the mean is computed for the
       flattened array by default

       Parameters
       ----------
       elements: flattened list or array

        Returns
       -------
       The mean value

       Example:
       array = [86, 87, 88, 86, 87, 85, 86]
       std = meanCalualation(array)
       output: 86.42857142857143
       """

    total_value = len(list_data)
    addition = sum(list_data)
    mean_value = addition / total_value
    return mean_value


def stdCalualation(elements=[]):
    """"
    Compute the standard deviation

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default

    Parameters
    ----------
    elements: flattened list or array

     Returns
    -------
    The STD value

    Example:
    array = [86, 87, 88, 86, 87, 85, 86]
    std = stdCalualation(array)
    output: 0.9035079029052513
    """

    squre_sum = []
    n_elements = len(elements)
    for i in elements:
        normal = i - meanCalualation(elements)
        squre_sum.append(normal ** 2)
    number = sum(squre_sum) / n_elements
    return number ** (1 / 2)

'''
stdCalualation() accept array or list which is flatten by default
'''

std = stdCalualation(speed)
print("STANDARD DEVIATION by calculation: ", std)
