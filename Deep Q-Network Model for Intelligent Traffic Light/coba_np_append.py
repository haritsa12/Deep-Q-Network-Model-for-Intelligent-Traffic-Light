import numpy as np 
import math
# a = np.array([[1,2,3],[4,5,6]]) 

# print ('First array:' )
# print (a) 
# print ('\n')  

# print ('Append elements to array:') 
# print (np.append(a, [7,8,9])) 
# print ('\n' ) 

# print ('Append elements along axis 0:') 
# print (np.append(a, [[7,8,9]]) )
# print ('\n')  

# print ('Append elements along axis 1:') 
# print (np.append(a, [[5,5,5],[7,8,9]]))
timings = np.random.weibull(2, 10)
timings = np.sort(timings)

car_gen_steps=[]
min_old = math.floor(timings[0])
max_old = math.ceil(timings[-1])
min_new = 0
max_new = 10
for value in timings:
    car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

car_gen_steps = np.rint(car_gen_steps)

print(car_gen_steps)

a=np.random.randint(1, 2)
print (a)