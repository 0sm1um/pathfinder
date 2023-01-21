from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np


def generate_dataset():
    grid = specify_map(num_tiles = 3)
    
    start = grid.node(0, 0)
    end = grid.node(5, 5)
    finder = AStarFinder()
    path, runs = finder.find_path(start, end, grid)
    
    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start, end=end))
    
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
    grid : pathfinding.grid
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
    grid = Grid(matrix=matrix)
    return grid
def generate_trajectory(start, end):
    pass

def check_legality():
    pass

generate_dataset()
