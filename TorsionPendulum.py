"""
This file contains a simulation of the torsion pendulum.

Alex Angus

April 23, 2020
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

class TorsionPendulum():
    """
    Parameter Storage, model definitions, and solving function.
    """
    def __init__(self, plot_pe=False):
        
        # damping/torsion constants
        self.J = 0.42366
        self.w0 = 2*np.pi*0.618#0.821817
        self.t0 = 2.4#2.359666

        self.r = 2*self.J/self.t0
        self.D = self.J*self.w0**2
        
        # drive constants
        self.drive_amplitude = 0.4
        self.drive_frequency = 0.6
        
        #weight constants
        self.mass = 0.4
        self.g = 9.8
        self.radius = 9.53e-1
        self.weight_angle = 0.15
        
        if plot_pe:
            self.potential_energy()
        
    def potential_energy(self):
        """
        Plots the potential energy of the system with the current parameters.
        """
        pe = lambda theta : 0.5*self.D*theta**2 + \
        self.radius*self.g*2*(self.mass*np.cos(theta+self.weight_angle)+
                            self.mass*np.cos(theta-self.weight_angle))
        angles = np.linspace(-np.pi, np.pi, 100)
        pe_values = pe(angles)
        plt.plot(angles, pe_values, color='black')
        plt.ylabel(r"$U(\Theta)$")
        plt.xlabel(r"$\Theta$")
        plt.title("Potential Energy of Chaotic System")
        plt.show()
        
        
    def chaotic(self, initial_conditions, t):
        """
        Model that exhibits chaotic behavior -- with driving and weights 
        attached
        """
        y = initial_conditions[0]
        dy = initial_conditions[1]
        xdot = [[],[]]
        xdot[0] = dy
        xdot[1] = -(1/self.J)*(self.r*dy + self.D*(y + self.drive_amplitude*np.sin(
                2*np.pi*self.drive_frequency*t)) - (self.mass*self.g*
                self.radius*np.sin(y + self.weight_angle)) - (self.mass*self.g*
                self.radius*np.sin(y - self.weight_angle)))
    
        return xdot
    
    def damped_driven(self, initial_conditions, t):
        """
        The damped driven model -- with driving, but no weights
        """
        self.drive_amplitude = 0.3
        self.drive_frequency = 0.65
        y = initial_conditions[0]
        dy = initial_conditions[1]
        xdot = [[],[]]
        xdot[0] = dy
        xdot[1] = -(1/self.J)*(self.r*dy + self.D*(y + self.drive_amplitude*np.sin(
                2*np.pi*self.drive_frequency*t)))
    
        return xdot
    
    def damped(self, initial_conditions, t):
        """
        The simple harmonic oscillator model -- undriven, no weights, just friction
        """
        self.J = 0.42366
        self.w0 = 2*np.pi*0.618
        self.t0 = 2.4

        self.r = 2*self.J/self.t0
        self.D = self.J*self.w0**2
        y = initial_conditions[0]
        dy = initial_conditions[1]
        xdot = [[],[]]
        xdot[0] = dy
        xdot[1] = -(1/self.J)*(self.r*dy + self.D*y)
    
        return xdot

    def integrate(self, initial_conditions, times, model_type='damped'):
        """
        Uses numeric temporal discretization to approximate the solution to
        the equations of motion
        
        returns an array of angular displacement values and angular velocity
        values at discrete times
        """
        if model_type == 'damped':
            model = self.damped
        elif model_type == 'damped_driven':
            model = self.damped_driven
        else:
            model = self.chaotic
        phi_dot = np.transpose(odeint(model, initial_conditions, times))
        theta = phi_dot[0]
        theta_dot = phi_dot[1]
        
        return theta, theta_dot

#you should probably write your own code.