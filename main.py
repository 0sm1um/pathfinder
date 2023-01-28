from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np


def generate_dataset():
    matrix = tile_map(num_tiles = 6)
    
    intersections = labelIntersections(matrix)
        
    position_paths, models = generate_paths(matrix)
    
    print(intersections)
    print(position_paths[25])
    print(models[25])
    print('hello world')

def tile_map(num_tiles):
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
    position = []
    models = []
    boundary_index = matrix.shape[0]-1

    for i in range(boundary_index):
        start = grid.node(0,i) # Left Wall
        for j in range(boundary_index):
            grid.cleanup()
            end = grid.node(boundary_index,j) # Right Wall
            path,runs = finder.find_path(start,end,grid)
            # Generate path in opposite direction
            grid.cleanup()
            reversePath,runs = finder.find_path(end,start,grid)
            relativePath = coordinate2relative(path) # Map from coordinates to directional sequence
            # Legality Check goes here
            reverseRelativePath = coordinate2relative(reversePath)
            # Legality Check goes here
            position.append(path)
            position.append(reversePath)
            models.append(relative2motionmodel(relativePath))
            models.append(relative2motionmodel(reverseRelativePath))
    
    for i in range(boundary_index):
        start = grid.node(i,0) # Bottom Wall
        for j in range(boundary_index):
            grid.cleanup()
            end = grid.node(boundary_index,j) # Right Wall
            path,runs = finder.find_path(start,end,grid)
            # Generate path in opposite direction
            grid.cleanup()
            reversePath,runs = finder.find_path(end,start,grid)
            relativePath = coordinate2relative(path) # Map from coordinates to directional sequence
            reverseRelativePath = coordinate2relative(reversePath)
            # Legality Check goes here
            position.append(path)
            position.append(reversePath)
            models.append(relative2motionmodel(relativePath))
            models.append(relative2motionmodel(reverseRelativePath))

    ''' This is a block of code to print trajectories.
    for i in range(len(paths)):
        print(grid.grid_str(path=paths[i], start=start, end=end))
    print(len(paths))
    '''
    return position, models

def coordinate2relative(path):
    relativepath = []
    for i in range(1,len(path)):
        if np.array_equal((np.array(path[i])-np.array(path[i-1])),np.array([0,1])) == True:
            relativepath.append('Up')
        elif np.array_equal((np.array(path[i])-np.array(path[i-1])), np.array([0,-1])) == True:
            relativepath.append('Down')
        elif np.array_equal((np.array(path[i])-np.array(path[i-1])), np.array([-1,0])) == True:
            relativepath.append('Left')
        else:
            relativepath.append('Right')
    return relativepath

def relative2motionmodel(path):
    for i in range(1,len(path)):
        # Check if a turn occured
        if path[i] != path[i-1]:
            if path[i-1] == 'Up' and path[i] == 'Right':
                path[i-1] = 'Right Turn'
            elif path[i-1] == 'Right' and path[i] == 'Down':
                path[i-1] = 'Right Turn'
            elif path[i-1] == 'Down' and path[i] == 'Left':
                path[i-1] = 'Right Turn'
            elif path[i-1] == 'Left' and path[i] == 'Up':
                path[i-1] = 'Right Turn'
            else:
                path[i-1] = 'Left Turn'
    return path

#def detectIntersections(models):
#    # Find a turn
#    pass

def labelIntersections(matrix):
    # First we need to label intersections.
    intersections = []
    boundary_index = matrix.shape[0]-1
    # Start by labelling the boundary.
    for i in range(boundary_index):
        if i % 6 == 0:
            intersections.append((0,i))
            intersections.append((i,0))
            intersections.append((boundary_index,i))
            intersections.append((i,boundary_index))
            # Once an intersection is identified on the boundary, iterate across
            for j in range(1,boundary_index):
                if j % 6 == 0 and i != 0:
                    intersections.append((j,i)) # Mark Intersections Horizontally
                    intersections.append((j-1,i)) # Mark Adjacent Points Horizontally
                    intersections.append((j+1,i)) # Mark Adjacent Point Horizontally
                    intersections.append((i,j-1))# Mark Adjacent Points Vertically
                    intersections.append((i,j+1)) # Mark Adjacent Point Vertically
    return list(set([i for i in intersections]))

def labelOneWayStreet(matrix):
    pass
def check_legality():
    pass

generate_dataset()
