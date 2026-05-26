import numpy as np
import math
import sys

from interpolators import fastInterp2 

x = np.array([0, 1])
y = np.array([0, 1])
z = np.array([[1, 2],[1, 2]])

# Check 2D interpolator with 11 cases

# Case 1 - Code 1,1
print(f"Case 1 - Code 1-1 outside x and y values.")
xi = -1.0
yi = -1.0
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1 should be returned.")

# Case 2 - Code 1,2 midpoint
print(f"Case 2 - Code 1-2 outside x and inside y values (on midpoint).")
xi = -1
yi = 0.5
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1 should be returned.")

# Case 3 - Code 1,2 ahead of midpoint
print(f"Case 3 - Code 1-2 outside x and inside y values (ahead of midpoint).")
xi = -1
yi = 0.51
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 2 should be returned.")

# Case 4 - Code 1,3
print(f"Case 4 - Code 1-3 below x and above y values.")
xi = -1
yi = 0.51
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 2 should be returned.")

# Case 5 - Code 2,1 midpoint
print(f"Case 5 - Code 2-1 midpoint x and below y values.")
xi = 0.50
yi = -1
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1 should be returned.")

# Case 5 - Code 2,1 past midpoint
print(f"Case 5 - Code 2-1 midpoint x and below y values.")
xi = 0.51
yi = -1
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1 should be returned.")

# Case 6 - Code 2,2 
print(f"Case 6 - Code 2-2 within x and y values.")
xi = 0.50
yi = 0.50
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1.5 should be returned.")

# Case 7 - Code 2,3 
print(f"Case 7 - Code 2-3 within x and past y values.")
xi = 0.50
yi = 1.50
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 2.0 should be returned.")

# Case 8 - Code 3,1 
print(f"Case 8 - Code 3-1 ahead of x and before y values.")
xi = 1.50
yi = -1
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1.0 should be returned.")

# Case 9 - Code 3,2 midpoint
print(f"Case 9 - Code 3-2 ahead of x and within y values at midpoint.")
xi = 1.50
yi = 0.5
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 1.0 should be returned.")

# Case 10 - Code 3,2 past midpoint
print(f"Case 10 - Code 3-2 ahead of x and within y values past midpoint.")
xi = 1.50
yi = 0.51
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 2.0 should be returned.")

# Case 11 - Code 3,3 past midpoint
print(f"Case 10 - Code 3-3 ahead of x and ahead of y values.")
xi = 1.50
yi = 1.50
zi = fastInterp2(x,y,z,xi,yi)
print(f"    Interpolation gives {zi} and 2.0 should be returned.")