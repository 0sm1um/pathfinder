from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np


def generate_dataset():
    matrix = specify_map(num_tiles = 6)
    
    
    paths = generate_paths(matrix)
    
    'print paths'
    
    
    
    print('hello world')

def specify_map(num_tiles):
    '''
    This function is meant to specify the map which will be used for pathfinding.
    The general approach is to define a square tile which will be repeated
    into a larger space.

    Parameters
    ----------
    num_tiles : int
        The number of times this pattern tesselates.

    Returns
    -------
    matrix : numpy.array
        The "map" which is used for pathfinding.
    '''
    tile = np.array([
                     [1,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [1,0,0,0,0,0],
                     [1,1,1,1,1,1]
                     ])

    # Matrix Operations to define "large" space.
    
    matrix = np.concatenate([tile for e in range(num_tiles)])
    matrix = np.concatenate([matrix for e in range(num_tiles)],axis=1)

    # These final rows are added to make the map symmetrical.
    
    matrix = np.concatenate([np.ones([1,matrix.shape[0]]),matrix])
    matrix = np.concatenate([matrix,np.ones([matrix.shape[0],1])],axis=1)
    return matrix
def generate_paths(matrix):
    grid = Grid(matrix=matrix)
    finder = AStarFinder()
    paths = []
    boundary_index = matrix.shape[0]-1
    
    for i in range(boundary_index):
        grid.cleanup()
        start = grid.node(0,i) # Left Wall
        for j in range(boundary_index):
            grid.cleanup()
            end = grid.node(boundary_index,j) # Right Wall
            path, runs = finder.find_path(start,end,grid)
            # Map from coordinates to directional sequence
            # Legality Check goes here
            paths.append(path)
        
    for i in range(boundary_index):
        grid.cleanup()
        start = grid.node(i,0) # Bottom Wall
        for j in range(boundary_index):
            grid.cleanup()
            end = grid.node(j,boundary_index) # Top Wall
            path, runs = finder.find_path(start,end,grid)
            # Map from coordinates to directional sequence
            # Legality Check goes here
            paths.append(path)
    
    ''' This is a block of code to print trajectories.
    for i in range(len(paths)):
        print(grid.grid_str(path=paths[i], start=start, end=end))
    print(len(paths))
    '''
    return paths


def check_legality():
    pass

generate_dataset()
