"""
Author: Kai Chen(kchen916@connect.hkust-gz.edu.cn)
"""

from collections import defaultdict, deque
from obstacle import is_seg_valid_cspace, get_joint_index
from kdtree import KDTree
import numpy as np
import heapq
import sys
import time
import itertools

joint_step_size = 0.1

class Graph:
    """
    Graph for PRM
    """
    def __init__(self, dim, capacity=1000000):
        self._edges = defaultdict(list)
        self._kd = KDTree(dim, capacity)
        # start_id is the start node id, it will change when we call find_path
        self.start_id = None
        self.target_id = None

    def __len__(self):
        return len(self._kd)

    def insert_new_node(self, point):
        node_id = self._kd.insert(point)
        return node_id

    def add_edge(self, node_id, neighbor_id):
        """
        add edge between node_id and neighbor_id
        """
        self._edges[node_id].append(neighbor_id)
        self._edges[neighbor_id].append(node_id)

    def get_parent(self, child_id):
        return self._edges[child_id]

    def get_point(self, node_id):
        return self._kd.get_node(node_id).point

    def get_nearest_node(self, point):
        return self._kd.find_nearest_point(point)

    def get_neighbor_within_radius(self, point, radius):
        """
        return the neighbor node id within the radius
        """
        return self._kd.find_points_within_radius(point, radius)

def gen_prm_graph(arm, occupy_grid, graph, max_sample_nodes=5000, valid_neighbor_limit=20):
    """
    generte probability roadmap graph
    """
    print("[INFO] Generating probability roadmap graph")
    graph_node_list = []
    node_num = 0
    while len(graph) < max_sample_nodes:
        print('\r', end='')
        print("Processing: {:.2f}% ".format(len(graph)/max_sample_nodes*100), '▉'*int((len(graph)/max_sample_nodes)*100//3), end='')
        sys.stdout.flush()
        
        # [task 1] Construct the c-space roadmap, sample valid joint angles, and check they are collision free in c-space.
        # [hint] 1.use arm.sample_valid_joint() to sample a valid joint angle;
        #        2.use get_joint_index(joint_sampled) to get the joint index in the c-space grid;
        #        3.in c-space occupy_grid, 1 means collision, 0 means no collision;
        #        4.graph stores the sampled joint in the kdtree, use graph.insert_new_node(joint_sampled) to insert the new valid node;
        #        5.use graph.get_neighbor_within_radius(joint_sampled, radius) to get the ids of neighbor nodes within the radius
        #        6.use graph.get_point(neighbour_id) to get the point in c-space by node id
        #        7.use is_seg_valid_cspace(q0, q1, occupy_grid) to check if the line segment between two configuration is valid，then use graph.add_edge(parent_id, child_id) to add edge
        #        8.you can limit the number of neighbors for each node by valid_neighbor_limit
        # [task 1] 构造C空间路标图，采样有效关节角，并检查是否无碰撞
        # 使用arm.sample_valid_joint()采样关节角
        joint_sampled = arm.sample_valid_joint()

        # 获取关节角的网格索引
        joint_index = get_joint_index(joint_sampled)

        # 检查该关节角是否在占用网格中无碰撞（0表示无碰撞）
        if occupy_grid[joint_index[0], joint_index[1]] == 1:
            continue  # 存在碰撞，跳过该样本

        # 将无碰撞的关节角插入图中，获得节点ID
        node_id = graph.insert_new_node(joint_sampled)

        # 将节点添加到graph_node_list中，用于后续显示
        graph_node_list.append(joint_sampled)

        # 获取该节点的所有邻近节点ID（在指定半径内）
        radius = 0.5  # 邻域半径，可根据实际场景调整
        neighbor_ids = graph.get_neighbor_within_radius(joint_sampled, radius)

        valid_neighbors = 0  # 有效邻接节点计数器

        # 遍历所有邻近节点
        for neighbor_id in neighbor_ids:
            if valid_neighbors >= valid_neighbor_limit:
                break  # 达到邻居数量限制，停止添加
            
            # 获取邻近节点的关节角
            q_neighbor = graph.get_point(neighbor_id)
            
            # 检查当前节点与邻近节点之间的路径是否无碰撞
            if is_seg_valid_cspace(joint_sampled, q_neighbor, occupy_grid):
                # 添加无向边（双向连接）
                graph.add_edge(node_id, neighbor_id)
                valid_neighbors += 1

    print('\r', end='')
    time.sleep(0.1)
    return graph, graph_node_list

def find_path(arm, start_pos, goal_pos, graph: Graph, occupy_grid, joint_step_size=joint_step_size):
    s_theta = arm.inverse_kinematic(start_pos)
    g_theta = arm.inverse_kinematic(goal_pos)
    
    # add start and goal node to graph
    # use is_seg_valid_cspace to check if the edge is valid，then use graph.add_edge to add edge
    graph.start_id = graph.insert_new_node(s_theta)
    neighbor_ids = graph.get_neighbor_within_radius(s_theta, 0.5)  

    for neighbor_id in neighbor_ids:
            point_neighbor = graph.get_point(neighbor_id)
            if is_seg_valid_cspace(s_theta, point_neighbor, occupy_grid):
                graph.add_edge(graph.start_id, neighbor_id)
    
    graph.target_id = graph.insert_new_node(g_theta)
    neighbor_ids = graph.get_neighbor_within_radius(g_theta, 0.5)  
    for neighbor_id in neighbor_ids:
            point_neighbor = graph.get_point(neighbor_id)
            if is_seg_valid_cspace(g_theta, point_neighbor, occupy_grid):
                graph.add_edge(graph.target_id, neighbor_id)

    path = search(graph)
    path = smooth_path(path, joint_step_size)

    return deque(path), s_theta, g_theta

def search(graph):

    class cell:
        def __init__(self):
            self.g = float('inf')
            self.parent = -1

    road_graph = defaultdict(cell)
    road_graph[graph.start_id].g = 0

    found_path = False

    # [task 2] implement the search algorithm here
    # [hint] 1.start with graph.start_id, until find graph.target_id or no path found.
    #        2.use graph.get_parent(id) to get the ids of connected nodes of the node with id.
    #        3.use road_graph[id] to access the node to update the g_value, parent.
    # [task 2] 实现搜索算法
    # 初始化优先队列（按g值排序的堆）
    heap = []
    heapq.heappush(heap, (0, graph.start_id))  # (g_value, node_id)

    found_path = False  # 路径是否找到标志

    while heap:
        current_g, current_id = heapq.heappop(heap)  # 取出当前最小g值节点
        
        # 检查是否到达目标节点
        if current_id == graph.target_id:
            found_path = True
            break
        
        # 剪枝：如果当前路径代价大于已记录的最小代价，跳过该节点
        if current_g > road_graph[current_id].g:
            continue
        
        # 遍历当前节点的所有邻接节点
        for neighbor_id in graph.get_parent(current_id):
            # 获取两节点坐标计算欧氏距离
            q_current = graph.get_point(current_id)
            q_neighbor = graph.get_point(neighbor_id)
            distance = np.linalg.norm(q_current - q_neighbor)
            
            # 计算新路径代价
            tentative_g = road_graph[current_id].g + distance
            
            # 如果新路径更优，则更新邻接节点信息
            if tentative_g < road_graph[neighbor_id].g:
                road_graph[neighbor_id].g = tentative_g
                road_graph[neighbor_id].parent = current_id
                heapq.heappush(heap, (tentative_g, neighbor_id))  # 加入优先队列

    path = []
    if found_path:
        backward_path = [graph.get_point(graph.target_id)]
        node_id = road_graph[graph.target_id].parent
        while node_id != -1:
            backward_path.append(graph.get_point(node_id))
            node_id = (road_graph[node_id]).parent

        path = backward_path[::-1]

        print("[INFO PRM]: Found! Path length is {}. ".format(len(path)))
    else:
        print('[WARNNING PRM]: Unable to find a path!')

    return path

def smooth_path(path, joint_step_size=joint_step_size):
    path = [np.linspace(path[i], path[i + 1], int(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) / joint_step_size)) for i in range(len(path) - 1)]
    path = list(itertools.chain.from_iterable(path))

    return path