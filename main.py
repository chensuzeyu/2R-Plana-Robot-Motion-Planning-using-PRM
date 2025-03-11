from matplotlib.patches import Circle
from prm import Graph, gen_prm_graph, find_path, get_joint_index
from obstacle import gen_occupy_grid_name, gen_occupy_grid, occupy_grid_reso
from matplotlib.colors import from_levels_and_colors
from collections import deque
from arm import TwoLinkArm
from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import copy

WAIT_FOR_NEW_GOAL = 1
CAL_PATH_TO_GOAL = 2

# color param for display
colors = ['white', 'black']
levels = [0, 1, 2]
cmap, norm = from_levels_and_colors(levels, colors)

def is_goal_change(cur_goal, arm):
    """
    check if there is a new goal
    """
    new_goal = arm.goal
    arm_range = sum(arm.link_lengths)
    if arm.goal[0] is None or arm.goal[1] is None:
        return False, arm.joint_angles
     
    goal_dis = sqrt(arm.goal[0]**2+arm.goal[1]**2)
    if goal_dis > arm_range:
        print("[ERROR] Exceed the range of the manipulation")
        arm.goal = cur_goal
        return False, cur_goal

    diff = sqrt((cur_goal[0]-new_goal[0])**2+(cur_goal[1]-new_goal[1])**2)
    if diff > 1e-2:
        return True, new_goal
    else:
        return False, cur_goal

def draw_occupancy_grid(ax, s_theta, g_theta, graph_node_list, occupy_grid, route=None):
    """
    Draw the occupancy grid and the route
    """
    plt.sca(ax)
    s_index = get_joint_index(s_theta)
    g_index = get_joint_index(g_theta)
    plt.cla()
    c_circle = Circle((s_index[1], s_index[0]), 2, color='red')
    g_circle = Circle((g_index[1], g_index[0]), 2, color='green')
    ax.add_artist(c_circle)
    ax.add_artist(g_circle)

    plt.scatter(graph_node_list[:,1], graph_node_list[:,0], linewidths=0.01, alpha=0.1, color='blue')
    if route is not None:
        route = get_joint_index(route)
        plt.plot(route[:,1], route[:, 0], color='r')
    plt.matshow(occupy_grid, fignum=0, cmap=cmap)
    # ax.invert_yaxis()
    plt.xticks(range(0, occupy_grid.shape[1]+1, 25), [str('-$\pi$'), str('-$\pi$/2'), str('0'), str('$\pi$/2'), str('$\pi$')])
    plt.yticks(range(0, occupy_grid.shape[0]+1, 25), [str('-$\pi$'), str('-$\pi$/2'), str('0'), str('$\pi$/2'), str('$\pi$')])
    plt.draw()


def draw_init(ax, graph_node_list, occupy_grid):
    """
    """
    plt.sca(ax)
    plt.scatter(graph_node_list[:,1], graph_node_list[:,0], linewidths=0.01, alpha=0.1, color='blue')
    plt.matshow(occupy_grid, fignum=0, cmap=cmap)
    plt.xticks(range(0, occupy_grid.shape[1]+1, 25), [str('-$\pi$'), str('-$\pi$/2'), str('0'), str('$\pi$/2'), str('$\pi$')])
    plt.yticks(range(0, occupy_grid.shape[0]+1, 25), [str('-$\pi$'), str('-$\pi$/2'), str('0'), str('$\pi$/2'), str('$\pi$')])
    plt.draw()

def main():
    # Graph param
    capacity = 10000
   
    # PRM param
    sample_num = 2000
   
    # arm parma
    N_LINKS = 2
    link_lengths = [1] * N_LINKS
    start_pos = goal_pos = (0.0, 0.0)
    show_animation = True
   
    # obstacles param
    obstacles = [[1.5, 0.75, 0.6], [0, -1.6, 0.5]]
    # obstacles = [[1.5, 0.75, 0.6]]

    # instance arm and Graph
    arm = TwoLinkArm(link_lengths, start_pos, show_animation)
    graph = Graph(arm.n_links, capacity)
   
    # generate files to save the occupancy grid and the Graph
    # 1.generate name by Hash
    env_param = str(obstacles)+str(N_LINKS)+str(link_lengths)+str(occupy_grid_reso)+str(capacity)+str(sample_num)
    prefix = 'files'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    env_name, graph_name = gen_occupy_grid_name(env_param)
    env_name = os.path.join(prefix, env_name)
    graph_name = os.path.join(prefix, graph_name)
    
    # 2.check if files are existed or genereate them
    if os.path.isfile(env_name) :
        occupy_grid = pickle.load(open(env_name, 'rb'))
        print('[INFO] Load occupy_grid file, file name is {}'.format(env_name))
    else:
        print('[INFO] Not found occupy_grid, generate file and file_name is {}'.format(env_name)) 
        occupy_grid = gen_occupy_grid(arm, obstacles)
        with open(env_name, 'wb') as f:
            pickle.dump(occupy_grid, f, -1)      
    if os.path.isfile(graph_name):
        graph_data = pickle.load(open(graph_name, 'rb'))
        graph = graph_data[0]
        graph_node_list = graph_data[1]
        print('[INFO] Load graph file, file name is {}'.format(graph_name))
    else:
        print('[INFO] Not found graph, generate file and file_name is {}'.format(graph_name))
        graph, graph_node_list = gen_prm_graph(arm, occupy_grid, graph, sample_num)
        graph_node_list = np.array(graph_node_list)
        with open(graph_name, 'wb') as f:
            pickle.dump((graph, graph_node_list), f, -1)
              
    # a simple state machine, initial state
    state = WAIT_FOR_NEW_GOAL
   
    # initial some variables
    og_ax = arm.fig.add_subplot(121)
    cur_joint = arm.joint_angles
    goal_pos = arm.goal
    route = deque()
    first_time_display = True

    # MAIN LOOP
    while not arm.exit:
        is_changed, goal_pos = is_goal_change(goal_pos, arm)
        if is_changed:
            state = CAL_PATH_TO_GOAL
        if state is WAIT_FOR_NEW_GOAL:
            if len(route) != 0:
                cur_joint = route.popleft()
            arm.forward_kinematic(cur_joint, obstacles)
            start_pos = arm.end_effector
            s_theta = arm.joint_angles
            g_theta = s_theta
            if first_time_display:
                draw_init(og_ax, graph_node_list, occupy_grid)
                first_time_display = False
        elif state is CAL_PATH_TO_GOAL:
            print('[INFO] Assigned a new goal: ', goal_pos)
            route, s_theta, g_theta = find_path(arm, start_pos, goal_pos, graph, occupy_grid)  
            state = WAIT_FOR_NEW_GOAL
            if len(route):
                display_route = np.array(copy.copy(route))
            else:
                display_route = None
            draw_occupancy_grid(og_ax, s_theta, g_theta, graph_node_list, occupy_grid, display_route)
            
if __name__ == '__main__':
    main()