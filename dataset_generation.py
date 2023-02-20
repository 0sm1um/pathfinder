# If its stupid and it works, it probably wasn't stupid - Marcus Aurelius
# The devil is in the details - Dr. Ruixin Niu
#from .datasetDefinition import SimulatedTrajectoryDataset
from datetime import datetime, timedelta
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import pandas as pd


from stonesoup.types.state import State
from stonesoup.types.detection import Detection
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_dataset():
    matrix = tile_map(num_tiles = 6)
    intersection_locations = label_intersections(matrix,6)
    print('Intersections Labeled and Defined')
    one_way_labels = label_one_way_street(matrix)
    print('One Way Streets Labeled and Defined')
    long_position_paths, long_models = generate_long_paths(matrix,one_way_labels)
    intersection_paths, intersection_models = generate_turning_paths(matrix,intersection_locations,one_way_labels,1)
    print('Trajectory Data Generation Complete')
    n = 9
    print('Interpolation scheme set to split grid into '+str(n)+' distinct time steps.')
    raw_long_data = format_data(long_position_paths,long_models,intersection_locations,one_way_labels,n)
    intersection_data = format_data(intersection_paths,intersection_models,intersection_locations,one_way_labels,n)
    print('Ground Truth Data Formatted')
    long_ground_truth = interpolate_trajectories(raw_long_data,n,(0.005,0.025,2))
    intersections_ground_truth = interpolate_trajectories(intersection_data,n,(0.005,0.075,2))
    num_repetitions = int(sum([4*len(l) for l in long_ground_truth])/sum([len(l) for l in intersections_ground_truth]))
    for i in range(num_repetitions): # Generate turns until datasets are balanced
        intersections_ground_truth.extend(interpolate_trajectories(intersection_data,n,(0.005,0.075,2)))
    print('Ground Truth Data Interpolated')
    long_prediction_data = generate_predictions(long_ground_truth,n)
    intersections_prediction_data = generate_predictions(intersections_ground_truth,n)
    print('Predictions Generated')
    long_predictions_tensor = form_tensor(long_prediction_data,device)
    intersection_predictions_tensor = form_tensor(intersections_prediction_data,device)
    print('Pytorch Tensors Formed')
    write_dataset_to_file(long_predictions_tensor, 'full_data_tensor.pt')
    write_dataset_to_file(intersection_predictions_tensor, 'intersection_data_tensor.pt')
    print('Dataset written to files.')
# TODO "Balance Dataset" Module
#   Iterate around the intersections, generate paths which go through them
#   Stop when len(newData) == len(oldData) (a balanced dataset)
#   If that doesn't work, set start and endpoints to only be adjacent to intersection
#   If THAT doesn't work, try to apply Transfer Learning. Train on intersections first

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

def generate_long_paths(matrix,oneWayLabels):
    position = []
    models = []
    boundary_index = matrix.shape[0]-1
#    turn_threshold = 0
    for i in range(boundary_index):
        start = (0,i) # Left Wall
        for j in range(boundary_index):
            matrix = tile_map(num_tiles = 6)
            end = (boundary_index,j) # Right Wall
            path,relative_path = generate_legal_path(matrix,start,end,oneWayLabels)
#            if count_turns(relative2motionmodel(relativePath)) < turn_threshold:
#                continue
            matrix = tile_map(num_tiles = 6)
            reverse_path, reverse_relative_path = generate_legal_path(matrix,end,start,oneWayLabels)
            position.append(path)
            position.append(reverse_path)
            models.append(relative2motionmodel(relative_path))
            models.append(relative2motionmodel(reverse_relative_path))
            
    for i in range(boundary_index):
        start = (i,0) # Bottom Wall
        for j in range(boundary_index):
            matrix = tile_map(num_tiles = 6)
            end = (i,boundary_index) # Top Wall
            path,relative_path = generate_legal_path(matrix,start,end,oneWayLabels)
            #if count_turns(relative2motionmodel(relativePath)) < turn_threshold:
            #    continue
            matrix = tile_map(num_tiles = 6)
            reverse_path,reverse_relative_path = generate_legal_path(matrix,end,start,oneWayLabels)
            position.append(path)
            position.append(reverse_path)
            models.append(relative2motionmodel(relative_path))
            models.append(relative2motionmodel(reverse_relative_path))
    '''
     #This is a block of code to print trajectories.
    grid = Grid(matrix=matrix)
    for i in range(len(path)):
        print(grid.grid_str(path=position[i]))
    print(len(path))
    '''
    return position, models

def generate_turning_paths(matrix,intersection_locations,one_way_labels,magic_number):
    # Intersections are used to determine start and endpoints
    boundary_index = matrix.shape[0]-1
    position = []
    models = []
    for intersection in intersection_locations:
        if intersection[0] % 6 == 0 and intersection[1] % 6 == 0:
            upper_bound = intersection[1] + magic_number
            lower_bound = intersection[1] - magic_number
            left_bound = intersection[0] - magic_number
            right_bound = intersection[0] + magic_number
            points = []
            if upper_bound < boundary_index:
                points.append((intersection[0],upper_bound))
            if lower_bound > 0:
                points.append((intersection[0],lower_bound))
            if left_bound > 0:
                points.append((left_bound,intersection[1]))
            if right_bound < boundary_index:
                points.append((right_bound,intersection[1]))
            for j in points:
                for k in points:
                    if j == k:
                        continue
                    else:
                        matrix = tile_map(num_tiles = 6)
                        path,relativePath = generate_legal_path(matrix,j,k,one_way_labels)
                        if len(path) > 3:
                            continue
                        matrix = tile_map(num_tiles = 6)
                        reversePath,reverseRelativePath = generate_legal_path(matrix,k,j,one_way_labels)
                        position.append(path)
                        position.append(reversePath)
                        models.append(relative2motionmodel(relativePath))
                        models.append(relative2motionmodel(reverseRelativePath))
    return position, models

def coordinate2relative(path):
    relativepath = []
    for i in range(len(path)-1):
        if np.array_equal((np.array(path[i+1])-np.array(path[i])),np.array([0,1])) == True:
            relativepath.append('Up')
        elif np.array_equal((np.array(path[i+1])-np.array(path[i])), np.array([0,-1])) == True:
            relativepath.append('Down')
        elif np.array_equal((np.array(path[i+1])-np.array(path[i])), np.array([-1,0])) == True:
            relativepath.append('Left')
        else:
            relativepath.append('Right')
    relativepath.append(relativepath[-1])
    return relativepath

def relative2motionmodel(relativePath):
    models = [relativePath[0]]
    for i in range(1,len(relativePath)): # First entry has no direction and is labelled 'Start'
        if relativePath[i] != relativePath[i-1]: # Check if a turn occured
            if relativePath[i-1] == 'Up' and relativePath[i] == 'Right':
                models.append('Right Turn')
            elif relativePath[i-1] == 'Right' and relativePath[i] == 'Down':
                models.append('Right Turn')
            elif relativePath[i-1] == 'Down' and relativePath[i] == 'Left':
                models.append('Right Turn')
            elif relativePath[i-1] == 'Left' and relativePath[i] == 'Up':
                models.append('Right Turn')
            else:
                models.append('Left Turn') # 2 Denotes Left Turn
        else:
            models.append(relativePath[i])
    return models

def label_intersections(matrix,magic_number):
    # First we need to label intersections.
    intersections = []
    boundary_index = matrix.shape[0]-1
    # Start by labelling the boundary.
    for i in range(boundary_index):
        if i % magic_number == 0:
            intersections.append((0,i))
            intersections.append((i,0))
            intersections.append((boundary_index,i))
            intersections.append((i,boundary_index))
            # Once an intersection is identified on the boundary, iterate across
            for j in range(1,boundary_index):
                if j % magic_number == 0 and i != 0:
                    intersections.append((j,i)) # Mark Intersections Horizontally
                    intersections.append((j-1,i)) # Mark Adjacent Points Horizontally
                    intersections.append((j+1,i)) # Mark Adjacent Point Horizontally
                    intersections.append((i,j-1)) # Mark Adjacent Points Vertically
                    intersections.append((i,j+1)) # Mark Adjacent Point Vertically
    return list(set([i for i in intersections]))

def label_one_way_street(matrix):
    ''' Note that these streets were arbitrarily chosen. Choosing these
    4 streets were simply aesthetically pleasing.'''
    boundary_index = matrix.shape[0]-1
    oneWay = []
    for i in range(boundary_index):
        if i == 24:
            oneWay.append((24,i,'Down'))
            oneWay.append((i,12,'Left'))
            oneWay.append((12,i,'Up'))
            oneWay.append((i,24,'Right'))
    return oneWay

def generate_legal_path(matrix,start,end,oneWay):
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
                    if len(path) == 0:
                        return [i for i in range(5000)] # If a path cannot be found, return a path greater than 5000 to be discarded.
                    break
                else:
                    continue
                break
        pathIsLegal = True
    return path, relativePath

def count_turns(relativePath):
    num_turns = 0
    for i in range(len(relativePath)):
        if relativePath[i] == 'Left Turn' or relativePath[i] == 'Right Turn':
            num_turns += 1
    return num_turns

def format_data(position_paths,models,intersectionLabels,oneWayLabels,n):
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
            # Add Base Velocities and reformat Model List
            # Note: 0 is Constant Velocity Model
            # Note: 1 is Right Turn Model
            # Note: -1 is Left Turn Model
            if models[i][j] == 'Up':
                xvel.append(0)
                yvel.append(1/n)
                models[i][j] = 0
            elif models[i][j] == 'Down':
                xvel.append(0)
                yvel.append(-1/n)
                models[i][j] = 0
            elif models[i][j] == 'Left':
                xvel.append(-1/n)
                yvel.append(0)
                models[i][j] = 0
            elif models[i][j] == 'Right':
                xvel.append(1/n)
                yvel.append(0)
                models[i][j] = 0
            else: # For Turns
                 xvel.append(xvel[-1])
                 yvel.append(yvel[-1])
                 if models[i][j] == 'Right Turn':
                     models[i][j] = 1 # 1 Denotes Right Turn
                 else:
                     models[i][j] = 2 # 2 denotes left turn
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
                    
        rawData.append([(x[k],
             xvel[k],
             y[k],
             yvel[k],
             intersections[k],
             oneWayUp[k],
             oneWayRight[k],
             oneWayDown[k],
             oneWayLeft[k],
             models[i][k]) for k in range(len(x))])
        
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
    return rawData        

def interpolate_trajectories(formatted_data,n,noise):
    # It takes n seconds to travel from one grid tile to another. for v=1/n
    # Turns happen in ~0.78*n seconds due to radius of turn being ~0.78 the distance of grid
    # First state shares labels with current iteration
    # Second state shares labels with previous
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
    interpolated_data = []
    for i in range(len(formatted_data)):
        for j in range(1,len(formatted_data[i])):
            #q = [0.005, 0.005] # Old Fixed Noise Value
            q = np.abs(np.random.normal(loc=noise[0],scale=noise[1],size=noise[2])) # Random noise to prevent overtraining to one noise value.
            if formatted_data[i][j][9] == 0: #If not turning apply CV model
                xi, xivel, yi, yivel = _interpolate_CV(formatted_data[i][j][0],
                                                       formatted_data[i][j][1],
                                                       formatted_data[i][j][2],
                                                       formatted_data[i][j][3],
                                                       q,
                                                       n)
            else:
                xi, xivel, yi, yivel = _interpolate_turn(formatted_data[i][j][0],
                                                         formatted_data[i][j][1],
                                                         formatted_data[i][j][2],
                                                         formatted_data[i][j][3],
                                                         formatted_data[i][j][9],
                                                         q,
                                                         n)
            x.extend(xi)
            xvel.extend(xivel)
            y.extend(yi)
            yvel.extend(yivel)
            for k in range(len(xi)):
                intersections.append(formatted_data[i][j][4])
                oneWayUp.append(formatted_data[i][j][5])
                oneWayRight.append(formatted_data[i][j][6])
                oneWayDown.append(formatted_data[i][j][7])
                oneWayLeft.append(formatted_data[i][j][8])
                models.append(formatted_data[i][j][9])
        if len(x) == 0: # If 
            continue
        interpolated_data.append([(x[k],
                                 xvel[k],
                                 y[k],
                                 yvel[k],
                                 intersections[k],
                                 oneWayUp[k],
                                 oneWayRight[k],
                                 oneWayDown[k],
                                 oneWayLeft[k],
                                 models[k]) for k in range(len(x))])
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
    return interpolated_data

def _interpolate_CV(x,xvel,y,yvel,q,n):
    originalState = State(state_vector=np.array([x,xvel,y,yvel]))
    timediff = timedelta(seconds=1)
    q_x = q[0]
    q_y = q[1]
    transitionModel = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x),
                                                          ConstantVelocity(q_y)])
    states = [transitionModel.function(originalState, noise=True, time_interval=t*timediff) for t in range(1,n+1)]

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

def _interpolate_turn(x,vx,y,vy,turnDirection,q,n):
    # Propagate forward 3 timesteps
    timediff = timedelta(seconds=1)
    num_steps = int(np.ceil(np.pi/4*n)) # Ratio of distance traveled turning versus straight
    if turnDirection == 1:
        transitionModel = KnownTurnRate(turn_noise_diff_coeffs=q,turn_rate=-np.pi/2/num_steps) # Turn 90 degrees per num_steps time
    else:
        transitionModel = KnownTurnRate(turn_noise_diff_coeffs=q,turn_rate=np.pi/2/num_steps)
    originalState = State(state_vector = np.array([x,vx,y,vy]))
    states = [transitionModel.function(originalState, noise=True, time_interval=t * timediff) for t in range(1,num_steps+1)]
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

def generate_predictions(groundTruthData,n):#:,model_list):
    model_list = [CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                           ConstantVelocity(0.005)]),
                  KnownTurnRate(turn_noise_diff_coeffs=0.005,turn_rate=np.pi/2/int(np.ceil(np.pi/4*n))),
                  KnownTurnRate(turn_noise_diff_coeffs=0.005,turn_rate=-np.pi/2/int(np.ceil(np.pi/4*n))),
                  ]
    measurement_model = LinearGaussian(ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
                                       mapping=(0, 2),  # Mapping measurement vector index to state index
                                       noise_covar=np.array([[0.25, 0],  # Covariance matrix for Gaussian PDF
                                                             [0, 0.25]])
                                       )
    initial_timestamp = datetime(2023, 2, 9, 15, 44)
    timediff = timedelta(seconds=1)
    predData = []
    for i in range(len(groundTruthData)):
        xCV = []
        xvCV = []
        yCV = []
        yvLT = []
        xLT = []
        xvLT = []
        yLT = []
        yvLT = []
        xRT = []
        xvRT = []
        yRT = []
        yvRT = []
        xmeas = []
        ymeas = []
        intersections = []
        oneWayUp = []
        oneWayRight = []
        oneWayDown = []
        oneWayLeft = []
        models = []
        state = State(state_vector = [groundTruthData[i][0][0],
                                      groundTruthData[i][0][1],
                                      groundTruthData[i][0][2],
                                      groundTruthData[i][0][3]],
                                      timestamp = initial_timestamp)
        measurement = measurement_model.function(state, noise=True)
        xmeas = [measurement[0]]
        ymeas = [measurement[1]]
        for j in range(len(groundTruthData[i])-1):
            state = State(state_vector = [groundTruthData[i][j][0],
                                          groundTruthData[i][j][1],
                                          groundTruthData[i][j][2],
                                          groundTruthData[i][j][3]],
                                          timestamp = initial_timestamp+timediff*j)
            predictions = [m.function(state=state,time_interval=timedelta(seconds=1)) for m in model_list]
            state = State(state_vector = [groundTruthData[i][j][0],
                                          groundTruthData[i][j][1],
                                          groundTruthData[i][j][2],
                                          groundTruthData[i][j][3]],
                                          timestamp = initial_timestamp+timediff*j)
            measurement = measurement_model.function(state, noise=True)
            xCV.append(predictions[0][0])
            xvCV.append(predictions[0][1])
            yCV.append(predictions[0][2])
            yvLT.append(predictions[0][3])
            xLT.append(predictions[1][0])
            xvLT.append(predictions[1][1])
            yLT.append(predictions[1][2])
            yvLT.append(predictions[1][3])
            xRT.append(predictions[2][0])
            xvRT.append(predictions[2][1])
            yRT.append(predictions[2][2])
            yvRT.append(predictions[2][3])
            xmeas.append(measurement[0])
            ymeas.append(measurement[1])
            intersections.append(groundTruthData[i][j][4])
            oneWayUp.append(groundTruthData[i][j][5])
            oneWayRight.append(groundTruthData[i][j][6])
            oneWayDown.append(groundTruthData[i][j][7])
            oneWayLeft.append(groundTruthData[i][j][8])
            models.append(groundTruthData[i][j][9])
        
        predData.append([(xCV[k],
                          xvCV[k],
                          yCV[k],
                          yvLT[k],
                          xLT[k],
                          xvLT[k],
                          yLT[k],
                          yvLT[k],
                          xRT[k],
                          xvRT[k],
                          yRT[k],
                          yvRT[k],
                          xmeas[k],
                          ymeas[k],
                          intersections[k],
                          oneWayUp[k],
                          oneWayRight[k],
                          oneWayDown[k],
                          oneWayLeft[k],
                          models[k]) for k in range(len(xCV))])
    return predData

def form_tensor(inputData,device):
    # This is the final step to feed trajectory data into pytorch dataset.
    # Unpack all trajectory prediction data into a BIG 2D tensor
    fullDataset = torch.tensor(inputData[0],dtype=torch.float)
    for i in range(1,len(inputData)):
        fullDataset = torch.cat((fullDataset,torch.tensor(inputData[i],dtype=torch.float)))
    return fullDataset

def write_dataset_to_file(full_dataset,name):
    pdFullData = pd.DataFrame(full_dataset.numpy())
    pdFullData.to_csv('full_data.csv',encoding='utf-8',index=False)
    torch.save(full_dataset, name)

tensor = generate_dataset()
