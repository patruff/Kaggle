import numpy

array = [.06, .25, 0, .15, .5, 0, 0, 0.04, 0, 0, None]
highCountX = 3
lowValY = .1

array_np = numpy.asarray(array)
print array_np

#array_np = array_np.fillna(0)

print "filled Na so ", array_np



low_values_indices = array_np < lowValY  # Where values are low
array_np[low_values_indices] = 0  # All low values set to 0

print "set low values to 0 ", array_np