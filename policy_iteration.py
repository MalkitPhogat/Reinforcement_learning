import numpy as np
import math

grid=np.zeros((3,4))
grid[1,1] = math.inf
grid[0,3], grid[1,3] = 1, -1

operation_grid = np.array([["R","R","R",1],
                         ["U",0,"U",-1],
                         ["U","R","U","L"]])

g=0.9
reward_next=0
print(operation_grid)
print(grid)
old_grid=grid.copy()
new_grid=np.zeros((3,4))
grid_data=np.zeros((4,3,4))
grid_data
y=0
while y<4:
    # new_grid=grid_data[y].copy()
    for i in range(0,3):
        for j in range(0,4):
            if(operation_grid[i][j]=="R"):
                new_grid[i][j] = old_grid[i][j+1] + g*new_grid[i][j+1]
            elif(operation_grid[i][j]=="U"):
                new_grid[i][j] = old_grid[i-1][j] + g*new_grid[i-1][j]
            elif(operation_grid[i][j]=="L"):
                new_grid[i][j] = old_grid[i][j-1] + g*new_grid[i][j-1]
            else:
                continue
    print()
    print(new_grid)
    # new_grid_2= np.newaxis(new_grid)
    # grid_data=np.append(grid_data, new_grid, axis=0)
    grid_data[y]=new_grid
    delta=grid_data[y]
    y=y+1