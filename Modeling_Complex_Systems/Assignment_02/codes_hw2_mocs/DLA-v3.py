#!/usr/bin/env python
# -*- coding: utf-8 -*-

#David W. Landay
#Last Updated: 10-10-2018

import numpy as np
from random import randint,seed
import pygame
from pygame import gfxdraw
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def update(particle, prob1, prob2, speed):
    """
        UPDATE: updates the position of the particle on the grid with
                probability prob

                ___ARGS___:

                particle: a Particle() object

                prob: a matrix describing the probability of moving (l, r, u, d)

                speed: rate of particle motion on grid. Type: Integer
    """
    if not prob2:
        #choose a direction to head in (choose left and right and an up and down):
        direction = speed * np.random.choice([-1, 1], p=prob1, replace=True, size=2)

        #update the coordinates (x, y):
        particle.pos = [particle.pos[0] + direction[0],
                        particle.pos[1] + direction[1]]
    else:
        direction_1 = speed * np.random.choice([-1, 1], p=prob1, replace=True)
        direction_2 = speed * np.random.choice([-1, 1], p=prob2, replace=True)

        #update the coordinates (x, y):
        particle.pos = [particle.pos[0] + direction_1,
                        particle.pos[1] + direction_2]


def circle(radius, width, height):
    """
        CIRCLE: to speed up DLA, circle() initializes a grid object by creating
                a circle centered about the middle pixel in the grid, of a
                specified radius

                ___ARGS___:

                radius: a value describing the distance away any fixed particle
                        lies from the center of a grid at initialization.
                        Type: Integer

                width: the width of the grid object. Type: Integer

                height: the height of the grid object. Type: Integer

                color: an rgba color for pygame to recognize
    """

    #define the center of the grid:
    pos = (width/2, height/2)

    #get the points along the circumference of the circle:
    theta = np.linspace(0,2*np.pi,1000)

    x = radius*np.cos(theta) + pos[0]
    y = radius*np.sin(theta) + pos[1]

    #store the fixed particles (will be assigned to the grid's fixed particles):
    fixed = [Particle(pos=[int(i), int(y[idx])], stationary = True) for idx, i in enumerate(x)]
    return fixed
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class Particle(object):
    """
        PARTICLE: A cartesian coordinate representation of a particle in the
                  DLA system

                  ___ARGS___:

                  pos:  a position on the grid [x,y]. Type: list

                  stationary: indicates whether the particle is moving.
                              Type: Bool

                __METHODS__:

                get_pos: returns the position of the particle

                get_stationary: returns the value of self.stationary
    """
    def __init__(self, pos=[int(500/2), int(300/2)], stationary=False):

        #particles will have a position in cartesian space (init to center pos):
        self.pos = pos #assume wdth & hght are even (initializes with grid w&h)

        #particle position updating stops if it hits a stationary particle:
        self.stationary = stationary

        self.x = pos[0]
        self.y = pos[1]

        def get_pos(self):
            return self.pos
        def get_stationary(self):
            return self.stationary
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class Grid(object):
    """
        GRID: The space on which Particles move

            ___ARGS___:

            n: number of particles in system: Type: Integer

            steps: number of repeated simulations. Type: Int (Needs update)

            invis: determines whether the simulation should run without animating
                   Type: Bool

            go2fixed: bias towards the fixed point from any quadrant.
                      Type: 0<=Float<1

            speed: how many pixels to hop around each iteration. Type: Integer

            particles: list of moving particles. Type: List

            fixed: list of fixed particles. Type: List

            width: width of the grid in pixels. Type: Integer

            height: height of the grid in pixels. Type: Integer

            make_stationary: distance away a collision occurs. Type: Integer

            part_size: size to draw particles (in pixels). Type: Integer

            radius: radius of fixed points if initialized with a circle of points.
                    Tyoe: integer


    """
    def __init__(self, n=10, steps=1000, invis=False,
                 go2fixed=0.0, speed=1, particles=[], fixed=[],
                 width=640, height=480, make_stationary=1, part_size=1,
                 radius=100):

        #number of particles:
        self.n = int(n)

        #number of steps:
        self.steps = int(steps)

        #turn invisible particles on or off (default: False):
        self.invis = invis

        #"gravitational attraction" towards the fixed particle:
        self.go2fixed = float(go2fixed)

        #how fast should the simulation run?
        self.speed = float(speed)

        #what is the probability of moving in a certain direction?
        self.go2fixed = go2fixed

        #store the position information for each particle:
        self.particles = particles
        self.fixed = fixed

        #define the size of the grid (default: 640x480):
        self.width  = int(width)
        self.height = int(height)

        #determine a stationary criteria:
        self.make_stationary = int(make_stationary)

        #random  seed if necesary:
        self.part_size = part_size

        #define radius for circle iniialization:
        self.radius = radius

    def initialize_particles(self):
        """
            DOCSTRING
        """
        #store particles:
        particles = []
        fixed = []

        #initialize all other particles:
        for i in range(self.n):
            #create a new particle:
            p = Particle(pos=[randint(0, self.width), randint(0, self.height)])
            particles.append(p)

        #initialize the first fixed particle:
        tp = Particle(stationary=True)
        fixed.append(tp)

        return  particles, fixed

    def initialize_with_circle(self):
        """
            DOCSTRING
        """

        #store particles:
        particles = []
        fixed = circle(self.radius,self.width,self.height)

        #initialize all other particles:
        for i in range(self.n):
            #create a new particle:
            p = Particle(pos=[randint(0, self.width), randint(0,self.height)])
            particles.append(p)

        return  particles, fixed

    def move(self, surf, color):
        """
            MOVE: creates the particle motion on the grid

                  ___ARGS___:

                  surf: the surface to draw on. Type: Pygame Surface object

                  color: RGBA representation of a color. Type: RGBA
        """

        #update all particles that are not stationary:
        for p in self.particles[:-1]:

            #account for angular velocity:
            if self.go2fixed == 0:
                self.prob = None #equal opportunity to move in any direction
            else:
                self.prob = np.array([0.5 - go2fixed, 0.5 + go2fixed])#bias

            if not p.stationary:
                update(p, self.prob, self.speed)

                #correct pos if going off screen:
                if p.pos[0] > int(self.width):
                    p.pos[0] = 0
                elif p.pos[0] < 0:
                    p.pos[0] = int(self.width)
                elif p.pos[1] > int(self.height):
                    p.pos[1] = 0
                elif p.pos[1] < 0:
                    p.pos[1] = int(self.height)

            #make particle stationary if touches another stationary particle:
            for s in self.fixed:
                if abs(p.pos[0] - s.pos[0]) == self.make_stationary and abs(p.pos[1] - s.pos[1]) == self.make_stationary:
                    p.stationary = True
                    self.fixed.append(p)
                    try:
                        self.particles.remove(p)
                    except:
                        pass

                #bounce the particle away so that two don't occupy the same space:
                elif p.pos == s:
                    update(p, [0.7,0.3], self.speed)
            pygame.draw.rect(surf, color, pygame.Rect(p.pos[0], p.pos[1], self.part_size, self.part_size))

    def move_without_drawing(self):
        """
            MOVE_WITHOUT_DRAWING: does the simulation without pygame
        """
        #update all particles that are not stationary:
        for p in self.particles[:-1]:

            #account for angular velocity:
            if self.go2fixed == 0:
                self.prob1 = None #equal opportunity to move in any direction
                self.prob2 = None
            elif self.go2fixed != 0 and p.pos[0] - self.width/2 >= 0 and p.pos[1] - self.height/2 >= 0: #p in quadrant 1:
                self.prob1 = np.array([0.5 + self.go2fixed, 0.5 - self.go2fixed])#bias
                self.prob2 = np.array([0.5 + self.go2fixed, 0.5 - self.go2fixed])
            elif self.go2fixed != 0 and p.pos[0] - self.width/2 <= 0 and p.pos[1] - self.height/2 >= 0: #p in quadrant 2:
                self.prob1 = np.array([0.5 - self.go2fixed, 0.5 + self.go2fixed])#bias
                self.prob2 = np.array([0.5 + self.go2fixed, 0.5 - self.go2fixed])
            elif self.go2fixed != 0 and p.pos[0] - self.width/2 <= 0 and p.pos[1] - self.height/2 <= 0: #p in quadrant 3:
                self.prob1 = np.array([0.5 - self.go2fixed, 0.5 + self.go2fixed])#bias
                self.prob2 = np.array([0.5 - self.go2fixed, 0.5 + self.go2fixed])
            elif self.go2fixed != 0 and p.pos[0] - self.width/2 >= 0 and p.pos[1] - self.height/2 <= 0: #p in quadrant 4:
                self.prob1 = np.array([0.5 + self.go2fixed, 0.5 - self.go2fixed])#bias
                self.prob2 = np.array([0.5 - self.go2fixed, 0.5 + self.go2fixed])

            if not p.stationary:
                update(p, self.prob, self.speed)

                #correct pos if going off screen:
                if p.pos[0] > int(self.width):
                    p.pos[0] = 0
                elif p.pos[0] < 0:
                    p.pos[0] = int(self.width)
                elif p.pos[1] > int(self.height):
                    p.pos[1] = 0
                elif p.pos[1] < 0:
                    p.pos[1] = int(self.height)

            #make particle stationary if touches another stationary particle:
            for s in self.fixed:
                if abs(p.pos[0] - s.pos[0]) == self.make_stationary and abs(p.pos[1] - s.pos[1]) == self.make_stationary:
                    p.stationary = True
                    self.fixed.append(p)
                    try:
                        self.particles.remove(p)
                    except:
                        pass

                #bounce the particle away so that two don't occupy the same space:
                elif p.pos == s:
                    update(p, [0.7,0.3], self.speed)

    def make_sim(self):
        self.initialize_with_circle()
        for s in range(self.steps):
            try:
                self.move_without_drawing()
            except:
                pass
        return self.fixed, self.particles


    def reset(self, surf):
        """
            Rewrites the pygame screen
        """
        surf.fill((0,0,0))

    def draw_sim_circle(self):
        """
            DRAW_SIM_CIRCLE: animates particles on the grid with pygame with an
                             a set of initial fixed points around a circle
        """
        #create a window to view simulation:
        pygame.init()
        simulation_window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Diffusion-Limited Aggregation Simulation")

        clock=pygame.time.Clock()

        #define colors for stuck and moving particles:
        stuck_color  = pygame.Color(255,62,150,255)
        moving_color = pygame.Color(255,255,255,255)

        #indicate whether the simulation is running or not:
        sim_active = True

        while sim_active == True:

            #quit if the red x is clicked:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sim_active = False



        pygame.quit()
        quit()
    def draw_sim(self):
        """
            DRAW_SIM: animates particles on the grid with pygame
        """
        #create a window to view simulation:
        pygame.init()
        simulation_window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Diffusion-Limited Aggregation Simulation")

        clock=pygame.time.Clock()

        #define colors for stuck and moving particles:
        stuck_color  = pygame.Color(255,62,150,255)
        moving_color = pygame.Color(255,255,255,255)

        #indicate whether the simulation is running or not:
        sim_active = True

        while sim_active == True:

            #quit if the red x is clicked:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sim_active = False

                    #draw the particles:

            #hold a list of rectangles:
            # rect_holder = [pygame.draw.rect(simulation_window,
            #                                 moving_color,
            #                                 (i.pos[0], i.pos[1], 5, 5))
            #                for i in self.particles]
            self.reset(simulation_window)
            try:
                self.move(simulation_window, moving_color)
            except:
                pass

            #move the particles until they are all stationary:
            for idx, p in enumerate(self.particles):


                try:
                    simulation_window.fill(stuck_color, (tuple(self.fixed[idx].pos), (1, 1)))
                except:
                    pass

            pygame.display.update()

        pygame.image.save(simulation_window, 'particles Diffusion-Limited Aggregation Simulation.png')
        pygame.quit()
        quit()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ =='__main__':

    #create a grid:
    p = Particle()
    grid = Grid(n=500, width=500, height=300, make_stationary=1, part_size=3, speed=2)
    grid.particles, grid.fixed = grid.initialize_particles()
    grid.draw_sim()
    #plt.imshow(grid.fixed)
