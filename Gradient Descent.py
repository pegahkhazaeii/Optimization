# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:11:24 2020

@author: Pegah Khazaie
"""

#Gradient Discent Method

import numpy as np
cur_x = 3 # The algorithm starts at x=3
cur_y=3  #The algorithm starts at y=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x,y: (2*x-4*y,-4*x+10*y-4)#Gradient of our function (f=x*2-4*x*y+5*y^2-4*y+3)

while previous_step_size > precision and iters < max_iters:
    prev_x = cur_x  #Store current x value in prev_x
    prev_y = cur_y
    (cur_x,cur_y) = (cur_x,cur_y) -  np.asarray(rate) * df(prev_x,prev_y) #Grad descent
    previous_step_size = np.linalg.norm(tuple(i-j for i,j in zip((cur_x,cur_y) ,(prev_x,prev_y))))  #Change in x & y
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nX value is",(cur_x,cur_y)) #Print iterations
    
print("The local minimum occurs at", (cur_x,cur_y)) #The local minimum occurs at (3.999731966762098, 1.999888976997694)