"""
This file is used for:  
    Generating an environment with obstacles.
    Checking method between the arm and obstacle.
Author: Kai Chen(kchen916@connect.hkust-gz.edu.cn)
"""

from math import pi, sqrt
import numpy as np
import hashlib
import sys

occupy_grid_reso = 100

def gen_occupy_grid_name(env_param):
    """
    according to parameters to generate file name of occupy grid with obstacle.
    In order to save time, we can save the occupy grid with obstacle.
    """
    md5 = hashlib.md5(env_param.encode('utf-8'))
    env_name = md5.hexdigest()+'_env.pickle'
    graph_name = md5.hexdigest()+'_graph.pickle'

    return env_name, graph_name

def gen_occupy_grid(arm, obstacles, resolution=occupy_grid_reso):
    """
    generate a grid with obstacle, 1 means collision, 0 means no collision.
    """
    print("[INFO] generating occupancy grid")
    grid = [[0 for _ in range(resolution)] for _ in range(resolution)]
    theta_list = [2*i*pi/resolution for i in range(-resolution//2, resolution//2+1)]

    for row in range(resolution):
        print('\r', end='')
        print("Processing: {:.2f}% ".format(row/resolution*100), '▉'*int((row/resolution)*100//3), end='')
        sys.stdout.flush()
        for col in range(resolution):
            arm.forward_kinematic([theta_list[row], theta_list[col]], show_animation=False)
            points = arm.points
            # arm.plot(obstacles) # for debug
            
            is_collision = False
            for i in range(len(points)-1):
                line_seg = [points[i], points[i + 1]]
                for obstacle in obstacles:
                    is_collision = not is_seg_valid(line_seg, obstacle)
                    if is_collision:
                        break
                if is_collision:
                    break
            grid[row][col] = int(is_collision)
    print('\r', end='')
    return np.array(grid)

def point_circle(point, c_point, r):
    """
    check if the point in the circle
    point: it can be any point around the circle or in the circle
    c_point: the center point of the circle
    r: radius of the circle
    """
    diff = c_point - point
    dis = sqrt(diff[0]**2 + diff[1]**2)

    if dis <= r:
        return True
    else:
        return False

def is_seg_valid(line_seg, obstacle):
    """
    check collision between line segment and obstacle(circle).
    """
    q0 = np.array(line_seg[0])
    q1 = np.array(line_seg[1])
    c = obstacle[0:2]
    r = obstacle[2]
    
    # check if in the circle
    if(point_circle(q0, c, r) or point_circle(q1, c, r)):
        return False
    
    # check line if in the circle
    line_vec = q1 - q0
    circle_vec = c - q0
    line_mag = np.linalg.norm(line_vec)
    proj = circle_vec.dot(line_vec / line_mag)
    if proj <= 0:
        closet_point = q0
    elif proj >= line_mag:
        closet_point = q1
    else:
        closet_point = q0 + line_vec * proj / line_mag
    if(point_circle(closet_point, c, r)):
        return False
    
    return True

def is_seg_valid_cspace(q0, q1, occupy_grid, interpolate_step=0.05):
    """
    check collision between line segment and obstacle(circle) in configuration space.
    """
    q0 = np.array(q0)
    q1 = np.array(q1)
    qs = np.linspace(q0, q1, int(np.linalg.norm(q1 - q0) / interpolate_step))
    for q in qs:
        joint_index = get_joint_index(q)
        if occupy_grid[joint_index[0], joint_index[1]] == 1:
            return False
    return True

def get_joint_index(joints_theta, resolusion=occupy_grid_reso):
    """
    get the index of joint in the grid.
    """
    joints_theta = np.array(joints_theta)
    index = np.array(joints_theta+pi)*resolusion/(2*pi)
    index = np.array(index, dtype=int)-1
    return index