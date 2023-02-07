from datetime import timedelta
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from stonesoup.types.state import State
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

import numpy as np

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_dataset():
    matrix = tile_map(num_tiles = 6)
    intersectionLabels = labelIntersections(matrix)
    oneWayLabels = labelOneWayStreet(matrix)
    position_paths, models = generate_paths(matrix,oneWayLabels)
    print('Trajectory Data Complete')
    rawData = formatData(position_paths,models,intersectionLabels,oneWayLabels)
    print('Ground Truth Data Formatted')
    interp_data = interpolateTrajectories(rawData)
    print('Grid Interpolated')
    #formTensor(rawData,device)
    return rawData

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

def generate_paths(matrix,oneWayLabels):
    position = []
    models = []
    boundary_index = matrix.shape[0]-1

    for i in range(boundary_index):
        start = (0,i) # Left Wall
        for j in range(boundary_index):
            matrix = tile_map(num_tiles = 6)
            end = (boundary_index,j) # Right Wall
            path,relativePath = generateLegalPath(matrix,start,end,oneWayLabels)
            reversePath,reverseRelativePath = generateLegalPath(matrix,end,start,oneWayLabels)
            position.append(path)
            position.append(reversePath)
            models.append(relative2motionmodel(relativePath))
            models.append(relative2motionmodel(reverseRelativePath))
            
    for i in range(boundary_index):
        start = (i,0) # Bottom Wall
        for j in range(boundary_index):
            matrix = tile_map(num_tiles = 6)
            end = (i,boundary_index) # Top Wall
            path,relativePath = generateLegalPath(matrix,start,end,oneWayLabels)
            reversePath,reverseRelativePath = generateLegalPath(matrix,end,start,oneWayLabels)
            position.append(path)
            position.append(reversePath)
            models.append(relative2motionmodel(relativePath))
            models.append(relative2motionmodel(reverseRelativePath))

    ''' This is a block of code to print trajectories.
    for i in range(len(paths)):
        print(grid.grid_str(path=position[i]))
    print(len(paths))
    '''
    return position, models

def coordinate2relative(path):
    relativepath = ['Start']
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

def relative2motionmodel(relativePath):
    for i in range(2,len(relativePath)): # First entry has no direction and is labelled 'Start'
        if relativePath[i] != relativePath[i-1]: # Check if a turn occured
            if relativePath[i-1] == 'Up' and relativePath[i] == 'Right':
                relativePath[i-1] = 'Right Turn'
            elif relativePath[i-1] == 'Right' and relativePath[i] == 'Down':
                relativePath[i-1] = 'Right Turn'
            elif relativePath[i-1] == 'Down' and relativePath[i] == 'Left':
                relativePath[i-1] = 'Right Turn'
            elif relativePath[i-1] == 'Left' and relativePath[i] == 'Up':
                relativePath[i-1] = 'Right Turn'
            else:
                relativePath[i-1] = 'Left Turn'
    return relativePath

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
                    intersections.append((i,j-1)) # Mark Adjacent Points Vertically
                    intersections.append((i,j+1)) # Mark Adjacent Point Vertically
    return list(set([i for i in intersections]))

def labelOneWayStreet(matrix):
    ''' Note that these streets were arbitrarily chosen. Choosing these
    4 streets were simply aesthetically pleasing.'''
    boundary_index = matrix.shape[0]-1
    oneWay = []
    for i in range(boundary_index):
        oneWay.append((24,i,'Down'))
        oneWay.append((i,12,'Left'))
        oneWay.append((12,i,'Up'))
        oneWay.append((i,24,'Right'))
    return oneWay

def generateLegalPath(matrix,start,end,oneWay):
    pathIsLegal = False
    grid = Grid(matrix=matrix)
    finder = AStarFinder()
    path,runs = finder.find_path(grid.node(start[0],start[1]),grid.node(end[0],end[1]),grid) # Find Initial Path
    relativePath = coordinate2relative(path)
    oneWayCoordinates = [(e[0],e[1]) for e in oneWay]
    oneWayDirections = [e[2] for e in oneWay]
    while pathIsLegal == False: # Legality Check
        for i in range(len(path)):
            for j in range(len(oneWayCoordinates)):
                if path[i] == oneWayCoordinates[j] and oneWayDirections[j] == relativePath[i]: # Check if target travel in prohibited direction
                    grid.cleanup()
                    matrix[path[i][0]][path[i][1]] = 0 # Add obstacle
                    grid = Grid(matrix=matrix)
                    path,runs = finder.find_path(grid.node(start[0],start[1]),grid.node(end[0],end[1]),grid) # Find New Path
                    relativePath = coordinate2relative(path)
                    break
                else:
                    continue
                break
        pathIsLegal = True
    return path, relativePath

def formatData(position_paths,models,intersectionLabels,oneWayLabels):
    rawData = []
    intersections = []
    x = []
    xvel = []
    y = []
    yvel = []
    oneWay = []
    oneWayUp = []
    oneWayRight = []
    oneWayDown = []
    oneWayLeft = []
    oneWayCoordinates = [(e[0],e[1]) for e in oneWayLabels]
    oneWayDirections = [e[2] for e in oneWayLabels]
    for i in range(len(position_paths)): # Iterate through each Path
        for j in range(len(position_paths[i])): #Check each point in trajectory
            x.append(position_paths[i][j][0])
            y.append(position_paths[i][j][1])
            if models[i][j] == 'Up':
                xvel.append(0)
                yvel.append(1/5)
            elif models[i][j] == 'Down':
                xvel.append(0)
                yvel.append(-1/5)
            elif models[i][j] == 'Left':
                xvel.append(-1/5)
                yvel.append(0)
            elif models[i][j] == 'Right':
                xvel.append(1/5)
                yvel.append(0)
            else: # For Turns and Initial Position
                 xvel.append('N/A')
                 yvel.append('N/A')
            for k in range(len(intersectionLabels)):
                if position_paths[i][j] == intersectionLabels[k]:
                    intersections.append(True)
                    break
                if k == len(intersectionLabels)-1:
                    intersections.append(False)
            for k in range(len(oneWayLabels)):
                if position_paths[i][j] == oneWayCoordinates[k]:
                    if oneWayDirections[k] == 'Up':
                        oneWayUp.append(True)
                        oneWayDown.append(False)
                        oneWayLeft.append(False)
                        oneWayRight.append(False)
                    elif oneWayDirections[k] == 'Down':
                        oneWayUp.append(False)
                        oneWayDown.append(False)
                        oneWayLeft.append(False)
                        oneWayRight.append(False)
                    elif oneWayDirections[k] == 'Left':
                        oneWayUp.append(False)
                        oneWayDown.append(False)
                        oneWayLeft.append(True)
                        oneWayRight.append(False)
                    elif oneWayDirections[k] == 'Right':
                        oneWayUp.append(False)
                        oneWayDown.append(False)
                        oneWayLeft.append(False)
                        oneWayRight.append(True)
                    break
                elif k == len(oneWayLabels)-1:
                    oneWay.append(False)
                    oneWayUp.append(False)
                    oneWayRight.append(False)
                    oneWayDown.append(False)
                    oneWayLeft.append(False)
                    
        rawData.append((x,
                        xvel,
                        y,
                        yvel,
                        intersections,
                        oneWayUp,
                        oneWayRight,
                        oneWayDown,
                        oneWayLeft,
                        models[i]))
        # Reset Loop Arrays
        x = []
        xvel = []
        y = []
        yvel = []
        intersections = []
        oneWayUp = []
        oneWayRight = []
        oneWayDown = []
        oneWayLeft = []
    
    #print(len(rawData[500][0]))
    #print(len(rawData[500][1]))
    #print(len(rawData[500][2]))
    #print(len(rawData[500][3]))
    #print(len(rawData[500][4]))
    #print(len(rawData[500][5]))
    #print(len(rawData[500][6]))
    #print(len(rawData[500][7]))
    #print(len(rawData[500][8]))
    #print(len(rawData[500][9]))
    #print(rawData[500])
    return rawData

def add_noise(rawData):
    for i in range(rawData):
        pass
        

def interpolateTrajectories(rawData):
    n = 5
    deltax = 1/n
    # n should equal 5
    # Add n-1 states "behind"
    # Turns happen in 4 seconds
    # First state shares labels with current iteration
    # Second state shares labels with previous
    interpolatedTrajectory = []
    x = []
    xvel = []
    y = []
    yvel = []
    intersections = []
    oneWayUp = []
    oneWayRight = []
    oneWayDown = []
    oneWayLeft = []
    models = []
    interpolatedData = []
    for i in range(len(rawData)):
        for j in range(1,len(rawData[i][9])):
            if rawData[i][9][j] != 'Left Turn' and rawData[i][9][j] != 'Right Turn': #If not turning apply CV model
                xi, xivel, yi, yivel = interpolateConstantVelocity(rawData[i][0][j],
                                                               rawData[i][1][j],
                                                               rawData[i][2][j],
                                                               rawData[i][3][j])
                x.extend(xi)
                xvel.extend(xivel)
                y.extend(yi)
                yvel.extend(yivel)
                for k in range(len(xi)):
                    intersections.append(rawData[i][4][j])
                    oneWayUp.append(rawData[i][5][j])
                    oneWayRight.append(rawData[i][6][j])
                    oneWayDown.append(rawData[i][7][j])
                    oneWayLeft.append(rawData[i][8][j])
                    models.append(rawData[i][9][j])
            else:
                try:
                    xi, xivel, yi, yivel = interpolateTurn(x[-1],
                                                       xvel[-1],
                                                       y[-1],
                                                       yvel[-1],
                                                       rawData[i][9][j])
                except IndexError:
                    break
                x.extend(xi)
                xvel.extend(xivel)
                y.extend(yi)
                yvel.extend(yivel)
                for k in range(len(xi)):
                    intersections.append(rawData[i][4][j])
                    oneWayUp.append(rawData[i][5][j])
                    oneWayRight.append(rawData[i][6][j])
                    oneWayDown.append(rawData[i][7][j])
                    oneWayLeft.append(rawData[i][8][j])
                    models.append(rawData[i][9][j])
                    
        interpolatedData.append((x,
                                 xvel,
                                 y,
                                 yvel,
                                 intersections,
                                 oneWayUp,
                                 oneWayRight,
                                 oneWayDown,
                                 oneWayLeft,
                                 models))
        #Reset Loop Arrays
        x = []
        xvel = [] 
        y = []
        yvel=[]
        intersections = []
        oneWayUp = []
        oneWayRight = []
        oneWayDown = []
        oneWayLeft = []
        models = []
    print(len(interpolatedData[500][0]))
    print(len(interpolatedData[500][1]))
    print(len(interpolatedData[500][2]))
    print(len(interpolatedData[500][3]))
    print(len(interpolatedData[500][4]))
    print(len(interpolatedData[500][5]))
    print(len(interpolatedData[500][6]))
    print(len(interpolatedData[500][7]))
    print(len(interpolatedData[500][8]))
    print(len(interpolatedData[500][9]))
    print(interpolatedData[500])
    return interpolatedData


def interpolateConstantVelocity(x,xvel,y,yvel):
    originalState = State(state_vector=np.array([x,xvel,y,yvel]))
    timediff = timedelta(seconds=1)
    q_x = 0.005
    q_y = 0.005
    transitionModel = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
    states = [transitionModel.function(originalState, noise=True, time_interval=-2*timediff),
              transitionModel.function(originalState, noise=True, time_interval=-1*timediff),
              originalState.state_vector,
              transitionModel.function(originalState, noise=True, time_interval=timediff),
              transitionModel.function(originalState, noise=True, time_interval=2*timediff)]
    x = []
    xvel = []
    y = []
    yvel = []
    for i in range(len(states)):
        x.append(states[i][0])
        xvel.append(states[i][1])
        y.append(states[i][2])
        yvel.append(states[i][3])
    return x, xvel, y, yvel

def interpolateTurn(xi,vxi,yi,vyi,turnDirection):
    # Propagate forward 3 timesteps
    x = []
    xvel = []
    y = []
    yvel = []
    q = [0.005, 0.005]
    timediff = timedelta(seconds=1)
    if turnDirection == 'Right Turn':
        transitionModel = KnownTurnRate(turn_noise_diff_coeffs=q,turn_rate=-np.pi/2/4)
    else:
        transitionModel = KnownTurnRate(turn_noise_diff_coeffs=q,turn_rate=np.pi/2/4)
    originalState = State(state_vector = np.array([xi,vxi,yi,vyi]))
    states = [transitionModel.function(originalState, noise=True, time_interval=timediff),
              transitionModel.function(originalState, noise=True, time_interval=2*timediff),
              transitionModel.function(originalState, noise=True, time_interval=3*timediff)]
    for i in range(len(states)):
        x.append(states[i][0])
        xvel.append(states[i][1])
        y.append(states[i][2])
        yvel.append(states[i][3])
    return x, xvel, y, yvel

def formTensor(rawData,device):
    
    pass

generate_dataset()
