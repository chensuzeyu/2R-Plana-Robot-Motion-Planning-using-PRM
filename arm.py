"""
Class for controlling and plotting an arm with an 2 number of links.
Author: Kai Chen(kchen916@connect.hkust-gz.edu.cn)
"""

from cmath import pi
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class TwoLinkArm(object):
    def __init__(self, link_lengths, init_ee_point, show_animation):
        self.show_animation = show_animation
        self.n_links = len(link_lengths)
        if self.n_links != 2:
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = self.inverse_kinematic(init_ee_point)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.goal = np.array(init_ee_point).T
        self.obstacles = []
        self.exit = False

        if show_animation: 
            self.fig = plt.figure('Arm', figsize=(11,5))
            self.ax = self.fig.add_subplot(1,2,2)
            self.fig.canvas.mpl_connect('button_press_event', self.click)
    
            plt.ion()
            plt.show()


    def inverse_kinematic(self, point):
        """
        Calculate the inverse kinematics for the desired point.
        """
        x, y = point[0], point[1]
        l1, l2 = self.link_lengths[0], self.link_lengths[1]

        c2 = (x**2+y**2-l1**2-l2**2)/(2*l1*l2)
        s2 = math.sqrt(1-c2**2)
        # solve the theta2 is always positive
        
        theta2 = math.atan2(s2, c2)

        k1 = l1+l2*c2
        k2 = l2*s2
        theta1 = math.atan2(y, x)-math.atan2(k2, k1)

        return np.array([theta1, theta2])

    def forward_kinematic(self, joint_angles, obstacles=[], show_animation=True):
        """
        Calculate the forward kinematics for the desired joint angles.
        """
        self.joint_angles = joint_angles
        self.update_ee_position(obstacles, show_animation)
    
    def sample_valid_joint(self):
        """
        Sample a valid joint angle.
        """
        q = np.random.random(self.n_links)
        return (q*2-1)*pi
    
    def update_ee_position(self, obstacles, show_animation=True):
        """
        Update the points in the arm.
        """
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T

        if show_animation:
            self.plot(obstacles)

    def key_event(self, event):
        if event.key == 'escape':
            self.exit = True
        
    def plot(self, obstacles=[]):  
        if len(obstacles) > 0:
            self.obstacles = obstacles
        plt.sca(self.ax)
        plt.cla()
        plt.axis('equal') 
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event', self.key_event)

        for obstacle in obstacles:
            circle = plt.Circle(
                (obstacle[0], obstacle[1]), radius=0.5*obstacle[2], fc='k')
            plt.gca().add_patch(circle)

        for i in range(self.n_links + 1):
            if i is not self.n_links:
                plt.plot([self.points[i][0], self.points[i + 1][0]],
                         [self.points[i][1], self.points[i + 1][1]], 'r-')
            plt.plot(self.points[i][0], self.points[i][1], 'k.')

        plt.text(-2.5, 3.3, 'click to give a new goal(Esc for exit)')
        reach_range = plt.Circle((0,0), sum(self.link_lengths), fill=False)
        plt.gca().add_artist(reach_range)
        limit_line = np.array([[-sum(self.link_lengths),0], [0,0]])
        plt.plot(limit_line[:, 0], limit_line[:,1], color='black')
        if self.goal[0] is not None and self.goal[1] is not None:
            plt.plot(self.goal[0], self.goal[1], 'gx')
        plt.plot([self.end_effector[0], self.goal[0]], [
                 self.end_effector[1], self.goal[1]], 'g--')

        plt.xlim([-self.lim-2, self.lim+2])
        plt.ylim([-self.lim-2, self.lim+2])
        plt.draw()
        if __name__ == '__main__':
            plt.pause(1)
        else:
            plt.pause(1e-4)
    

    def click(self, event):
        self.goal = np.array([event.xdata, event.ydata]).T

if __name__ == '__main__':
    # Simulation parameters
    dt = 0.1
    N_LINKS = 2
    N_ITERATIONS = 10000

    # States
    WAIT_FOR_NEW_GOAL = 1
    MOVING_TO_GOAL = 2

    show_animation = True

    link_lengths = [1] * N_LINKS
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = [N_LINKS, 0]
    arm = TwoLinkArm(link_lengths, joint_angles, show_animation)
    state = WAIT_FOR_NEW_GOAL

    arm.forward_kinematic(joint_angles)

