#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torsion Pendulum Chaotic motion analysis code

Created on Thu Apr 23 22:23:46 2020

@author: alexangus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from TorsionPendulum import TorsionPendulum
import nolds
from scipy.interpolate import interp2d

plt.rcParams.update({'font.size': 17}) #increase font size in plots

def clean_run(run, time):
    """
    Prepares run data for analysis
    
    params:
        run: array of floats representing angles of the pendulum
    
        time: array of floats representing times
    
    returns:
        cleaned_run: array of angles prepared for analysis
    
        starting_angle: the initial angle of the run
    
        time: the array of time values prepared for analysis
    """
    cleaned_run = [x for x in run if str(x) != 'nan']   # eliminate nan values from array
    cleaned_run -= np.mean(cleaned_run)                 # center data around 0
    if len(cleaned_run) != len(time):                   # ensure that dimensions of time and angle are same length
        time = time[:len(cleaned_run)]

    starting_angle = cleaned_run[0]                     # get initial angle
    return cleaned_run, starting_angle, time 

def read_excel(excelfile):
    """
    Reads an excel file of pendulum data and returns the times and angles of each run
    
    param:
        excelfile: the string name of the excel file where the data is located
    
    returns:    time array, and arrays of runs 1-3 (tuple of arrays)
    """
    df = pd.read_excel(excelfile, sheet_name='Sheet1')                       # read excel file in as pandas dataframe
    time = np.array(df['Time (s)'])                         # time array
    run1 = clean_run(np.array(df['Run 1']), time)            
    run2 = clean_run(np.array(df['Run 2']), time)
    run3 = clean_run(np.array(df['Run 3']), time)
    
    return run1, run2, run3       # return arrays excluding the first value of each (labels)

def make_damped_plots():
    """
    Creates a plot of damped, undriven oscillation. Compares measured
    data to simulation data
    """
    runs = read_excel('Undriven.xlsx')
    for run in runs:                                                            # for each set of data
        init_condit = [run[1], 0]                                               # get initial conditions
        tp = TorsionPendulum()                                                  # initialize model
        theta, theta_dot = tp.integrate(init_condit, run[2], model_type='damped')   # integrate model
        mse = str(round(np.sum((run[0] - theta)**2)/len(theta), 5))             # calculate the MSE
        
        fig = plt.figure(figsize=(7,6))                                         # make plot
        axis = fig.add_axes([0,0,1,1])
        axis.scatter(run[2], run[0], color='black', linewidth=3)
        axis.plot(run[2], theta, color='red')
        axis.set_title("Damped Oscillation: Simulation vs Data (MSE: {})".format(mse))
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Angular Displacement (rad)")
    
        axis.grid()
        ticks = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
        plt.yticks(ticks, ["-pi", "-2pi/3", "-pi/3", "0", "pi/3", "2pi/3", "pi"])
        extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("Damped_Simulation_MSE_{}.png".format(mse), bbox_inches=extent.expanded(1.4, 1.4))
        fig.show()
    
def make_driven_plots():
    """
    Creates a plot of damped driven oscillation. Compares measured
    data to simulation data
    """
    excelfile = 'Driven Data.xlsx'
    df = pd.read_excel(excelfile, sheet_name='Sheet2')                          # read excel file in as pandas dataframe
    time = np.array(df['Time (s)'])[:-46]                                       # time array
    run = np.array(df['Angle (rad)'])[:-46]                                     
    run -= np.mean(run)                                                         #center data around 0
    initial_angle = run[0]
    init_condit = [initial_angle, 0]
    tp = TorsionPendulum()
    theta, theta_dot = tp.integrate(init_condit, time, model_type='damped_driven')
    mse = str(round(np.sum((run - theta)**2)/len(theta), 5))
    fig = plt.figure(figsize=(7,6))
    axis = fig.add_axes([0,0,1,1])
    axis.scatter(time, run, color='black', linewidth=3)
    axis.plot(time, theta, color='red')
    axis.set_title("Damped Driven Oscillation: Simulation vs Data (MSE: {})".format(mse))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Angular Displacement (rad)")

    axis.grid()
    ticks = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    plt.yticks(ticks, ["-pi", "-2pi/3", "-pi/3", "0", "pi/3", "2pi/3", "pi"])
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("DampedDriven_Simulation_MSE_{}.png".format(mse), bbox_inches=extent.expanded(1.4, 1.4))
    fig.show()
    
def make_chaos_plot():
    """
    Creates a plot of simulated chaotic oscillation. 
    """
    time = np.linspace(0, 5000, 100000)
    init_angle = round(np.pi/3, 4) - 1e-4 - 1e-4
    init_condit = [init_angle, 0]
    tp = TorsionPendulum()
    theta, theta_dot = tp.integrate(init_condit, time, model_type='chaotic')
    fig = plt.figure(figsize=(17,6))
    axis = fig.add_axes([0,0,1,1])
    axis.plot(time, theta, color='black')
    axis.set_title("Chaotic Damped Driven Oscillation Simulation, Initial angle {}".format(init_angle))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Angular Displacement (rad)")

    axis.grid()
    ticks = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    plt.yticks(ticks, ["-pi", "-2pi/3", "-pi/3", "0", "pi/3", "2pi/3", "pi"])
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Chaotic Damped Driven Oscillation Simulation: theta0={}.png".format(init_angle), bbox_inches=extent.expanded(1.4, 1.4))
    fig.show()
    
def make_chaos_comparison_plot():
    """
    Compares several instances of chaotic motion with small variations in
    initial conditions
    """
    fig = plt.figure(figsize=(14,6))
    axis = fig.add_axes([0,0,1,1])
    time = np.linspace(0, 100, 2000)
    angles = []
    thetas = []
    for n in range(0, 18):
        initial_angle = round(np.pi/3 + n*1e-6, 6)                              # adjust initial condition of each instance
        angles.append(initial_angle)
        init = [initial_angle, 0]
        tp = TorsionPendulum()
        theta, theta_dot = tp.integrate(init, time, model_type='chaotic')
        thetas.append(theta)
        axis.plot(time, theta, linewidth=2.5)
    
    axis.set_title("Chaotic Displacement w Small Variation in Initial Conditions")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Angular Displacement (rad)")
    #axis.legend(angles, loc='best')
    axis.grid()
    ticks = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    plt.yticks(ticks, ["-pi", "-2pi/3", "-pi/3", "0", "pi/3", "2pi/3", "pi"])
    plt.ylim((-2*np.pi/3, 2*np.pi/3))
    #plt.xlim((40,60))
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Initial Conditions Comparison (large).png", bbox_inches=extent.expanded(1.4, 1.4))
    fig.show()
    return time, thetas

def make_chaos_simulation(time, thetas):
    """
    Produces images for an animation that compares several instances of chaotic
    motion with small changes in initial conditions.
    """
    for point in range(len(time)):
        fig = plt.figure(figsize=(7,7))
        axis = fig.add_subplot(111, projection='polar')
        axis.axis('off')
        for theta in thetas:
            axis.scatter(theta[point] + np.pi/2, 5, clip_on=False, s=100)
        fig.savefig("animation/{}.png".format(point))
            
        
def make_poincaret_plot():
    """
    Creates a 2D Poincaré plot from simulation data
    """
    init_condits = [np.pi/100, 0]
    num_points = 2e5
    total_time = 2e3
    take_point_every = 8
    sampling_frequency = total_time/num_points*take_point_every
    tp = TorsionPendulum()
    time = np.linspace(0, total_time, num_points)
    theta, theta_dot = tp.integrate(init_condits, time, model_type='chaotic')
    positions = []
    velocities = []
    count = 1
    for angle, velocity in zip(theta, theta_dot):
        if count%take_point_every == 0:
            positions.append(angle)
            velocities.append(velocity)
            count = 1
        else:
            count += 1
    fig = plt.figure(figsize=(7,6))
    axis = fig.add_axes([0,0,1,1])
    axis.scatter(positions, velocities, color='black', s=5)
    axis.grid()
    axis.set_title("Simulation of Angular Velocity vs Angular Displacement")
    axis.set_xlabel("Angular Displacement (rad)")
    axis.set_ylabel("Angular Velocity (rad/s)")
    axis.text(0, 2.2, "theta0: {} rad, f_sample: {} Hz \nf_drive: {}".format(
            round(init_condits[0], 3), round(sampling_frequency, 3),
            round(tp.drive_frequency, 3)))
    for position in positions:
        if position == np.any(positions):
            for velocity in velocities:
                if velocity == np.any(velocities):
                    print("Duplicate")
    """
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Poincaret Simulation: f_sample {}: theta0: {}.png".format(
            round(sampling_frequency, 3), round(init_condits[0], 4)), 
            bbox_inches=extent.expanded(1.4, 1.4))
    """
def make_3Dpoincaret_plot():
    """
    Creates a 3d Poincaré plot from simulation data
    """
    initial_angles = np.linspace(-np.pi, np.pi, 15)
    num_points = 2e4
    total_time = 2e3
    take_point_every = 8
    #sampling_frequency = total_time/num_points*take_point_every
    time = np.linspace(0, total_time, num_points)
    angles = []
    positions_3d = []
    velocities_3d = []
    for initial_angle in initial_angles:
        tp = TorsionPendulum()
        theta, theta_dot = tp.integrate([initial_angle, 0], time, model_type='chaotic')
        count = 1
        for angle, velocity in zip(theta, theta_dot):
            if count%take_point_every == 0:
                positions_3d.append(angle)
                velocities_3d.append(velocity)
                angles.append(initial_angle)
                count = 1
            else:
                count += 1
    fig = plt.figure(figsize=(20,20))
    axis = fig.add_subplot(111, projection='3d')
    axis.scatter(positions_3d, velocities_3d, angles, color='black', s=1)
    ticks = [-np.pi, -2*np.pi/3, -np.pi/3, 0, np.pi/3, 2*np.pi/3, np.pi]
    axis.set_xticks(ticks[1:-1])
    axis.set_xticklabels([r"$\frac{-2 \pi}{3}$", r"$\frac{-\pi}{3}$", "0", 
                          r"$\frac{\pi}{3}$", r"$\frac{2 \pi}{3}$"])   
    y_ticks = [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi]
    axis.set_yticks(y_ticks)
    axis.set_zticks(ticks)
    axis.set_zticklabels([r"$-\pi$", r"$\frac{-2 \pi}{3}$", r"$\frac{-\pi}{3}$",
                          "0", r"$\frac{\pi}{3}$", r"$\frac{2 \pi}{3}$", r"$\pi$"])
    axis.set_yticklabels([r"$-2 \pi$", r"$-\pi$", "0", r"$\pi$", r"$2 \pi$"])
    axis.set_title("Chaos with Different Iinital Conditions")
    axis.set_xlabel("$\Theta$ (rad)")
    axis.set_ylabel(r"$\frac{\partial \Theta}{\partial t}$ (rad/s)")
    axis.set_zlabel("$\Theta_0$ (rad)")
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Chaos with Different Initial Conditions", 
            bbox_inches=extent.expanded(1.4, 1.4))
    
#you should probably write your own code.
    
def plot_angle_fourier(tp, t, a, initial_angle):
    """
    Performs a fast fourier transform on angle vs time data and plots the
    resulting power spectrum
    """
    T = t[1] - t[0]     # sample spacing
    N = a.size
    
    frequency = np.linspace(0, 1/T, N)
    f_transform = np.abs(np.fft.fft(a))
    
    
    
    fig = plt.figure(figsize=(7,6))
    axis = fig.add_axes([0,0,1,1])
    axis.plot(frequency[:N//2], f_transform[:N//2]/N, color='black')
    axis.set_xlabel('Frequency (Hz)')
    axis.set_ylabel('Component Magnitude')
    axis.grid(alpha=0.5)
    axis.set_xlim((-0.1, 4))
    axis.set_yscale('log')
    axis.text(2, 0.5, r"$f_d = $ {} Hz".format(tp.drive_frequency))
    axis.set_title(r'Fourier Spectrum of Chaotic Displacement Signal. $\Theta_0 = ${}'.format(round(initial_angle, 4)))
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("FourierTransform_angle_time_{}.png".format(initial_angle), bbox_inches=extent.expanded(1.4, 1.4))
    fig.show()
    
def plot_angle_time(time, theta, initial_angle, velocity=None):
    """
    Plots angle vs time data and/or velocity vs time data
    """
    fig = plt.figure(figsize=(17,6))
    axis = fig.add_axes([0,0,1,1])
    axis.plot(time, theta, color='black')
    if velocity is not None:
        axis.plot(time, velocity, color='orange')
        axis.set_title("Anglular Displacement and Velocity")
        axis.legend([r"$\Theta$ (rad)", r"$\dot{\Theta}$ ($\frac{rad}{s}$)"])
    else:
        axis.set_title("Angular Displacement")
        axis.legend([r"$\Theta$"])
    axis.set_ylabel("Angle")
    axis.set_xlabel("Time (s)")
    axis.grid()
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Angle_Time_{}.png".format(initial_angle), bbox_inches=extent.expanded(1.4, 1.4))
        
def make_phase_plot(theta, theta_dot):
    """
    Creates a phase diagram of angular velocity vs angular displacement
    """
    fig = plt.figure(figsize=(7,6))
    axis = fig.add_axes([0,0,1,1])
    axis.grid()
    axis.plot(theta, theta_dot, color='brown', linewidth=0.2)
    axis.set_xlabel(r'$\Theta$ (rad)')
    axis.set_ylabel(r'$\dot{\Theta}$ (rad)')
    axis.set_title("Phase Plot ({})".format(r'$\Theta_0 = {}$'.format(
            round(theta[0], 4))))
    #fig.savefig("Phase_Plot_{}.png".format(r'$\Theta_0 = {}$'.format(
     #       round(theta[0], 4))))
    fig.show()

def chaos_report():
    """
    Calculates the maximum lyapunov exponent of the system for various parameters
    and initial conditions.
    """
    num_values = 1000                                                           # parameter spacing
    #theta0_values = np.linspace(-np.pi, np.pi, num_values)
    drive_frequencies = np.linspace(0, 2, num_values)
    #theta_dot0_values = np.linspace(-np.pi, np.pi, num_values)
    lines = []
    time = np.linspace(0, 100, 1000)
    
    for count, f in enumerate(drive_frequencies):
        print("Iteration: ", count)
        #for theta_dot in theta_dot0_values:
        tp = TorsionPendulum()
        tp.drive_frequency = f
        init_condits = [0, 0]
        chaotic_theta, chaotic_theta_dot = tp.integrate(init_condits, time, model_type='chaotic')
        driven_theta, driven_theta_dot = tp.integrate(init_condits, time, model_type='damped_driven')
        #damped_theta, damped_theta_dot = tp.integrate(init_condits, time, model_type='damped')
        chaotic_le = nolds.lyap_r(chaotic_theta, min_tsep = 10, lag = None)
        driven_le = nolds.lyap_r(driven_theta, min_tsep = 10, lag = None)
        #damped_le = nolds.lyap_r(damped_theta, min_tsep = 10, lag = None)
        lines.append([f, chaotic_le, driven_le])
    chaos_report = open("chaos_report_frequency_{}.txt".format(num_values), 'w')
    #header = "Theta\t Theta Dot\t Chaotic Lyapunov Exponent\t Damped Driven Lyapunov Exponent\t Damped Lyapunov Exponent\n"
    #header = "Theta\t Chaotic Lyapunov Exponent\t Damped Driven Lyapunov Exponent\t Damped Lyapunov Exponent\n"
    header = "Drive Frequency\t Chaotic Lyapunov Exponent\t Damped Driven Lyapunov Exponent\n"
    chaos_report.write(header)
    string_lines = []
    for line in lines:
        string_line = ""
        for value in line:
            string_line += (str(value) + '\t')
        string_lines.append(string_line + '\n')
    chaos_report.writelines(string_lines)
    chaos_report.close()
    
def make_2D_lyapunov_plots():
    """
    Creates a contour plot of maximum Lyapunov exponent vs 2 changing parameters
    for various models of the torsion pendulum
    """
    data = open("chaos_report_2D_10.txt", 'r')
    array = []
    for index, line in enumerate(data):
        if index != 0:
            values = line.split()
            array.append(values)
    data.close()
    columns = np.transpose(np.array(array, dtype=np.float64))
    colors = 'inferno'
    fig = plt.figure(figsize=(14,11))
    axis = fig.add_subplot(111)
    LE_function = interp2d(columns[0], columns[1], columns[2])                  # interpolate lyapunov data
    points = 1e2 #don't go over 1e3
    thetas = np.linspace(-np.pi, np.pi, points)
    theta_dots = np.linspace(-np.pi, np.pi, points)
    theta_grid, theta_dot_grid = np.meshgrid(thetas, theta_dots)
    LE_values = LE_function(thetas, theta_dots)
    normalization = matplotlib.colors.Normalize(vmin=min(LE_values.flatten()), vmax=max(LE_values.flatten()))
    fig.colorbar(cm.ScalarMappable(norm=normalization, cmap=colors))
    axis.contourf(theta_grid, theta_dot_grid, LE_values, 50, cmap=colors)
    axis.set_title("Max Lyapunov Exponent for Various Initial Conditions")
    axis.set_xlabel(r'$\Theta_0$ (rad)')
    axis.set_ylabel(r'$\dot{\Theta}_0$ (rad/s)')
    
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Lyapunov_Contour.png", bbox_inches=extent.expanded(1.4, 1.4))
        
    plt.show()
    fig = plt.figure(figsize=(14,11))
    axis = fig.add_subplot(111, projection='3d')
    normalization = matplotlib.colors.Normalize(vmin=min(LE_values.flatten()), vmax=max(LE_values.flatten()))
    fig.colorbar(cm.ScalarMappable(norm=normalization, cmap=colors))
    axis.plot_surface(theta_dot_grid, theta_grid, LE_values,
                      cmap=colors, edgecolor='none')
    axis.set_title("Max Lyapunov Exponent for Various Initial Conditions")
    
    axis.set_xlabel(r'$\Theta_0$ (rad)')
    axis.set_ylabel(r'$\dot{\Theta}_0$ (rad/s)')
    axis.set_zlabel(r'$\lambda_{\Theta}$', rotation=0)
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Lyapunov_3D.png", bbox_inches=extent.expanded(1.4, 1.4))
    plt.show()
   
    
def make_1D_lyapunov_plots():
    """
    Creates a plot of maximum lyapunov exponent vs 1 changing parameter for 
    various models of the torsion pendulum
    """
    data = open("chaos_report_frequency_1000.txt", 'r')
    theta_values = []
    chaos_values = []
    driven_values = []
    #damped_values = []
    for index, line in enumerate(data):
        if index != 0:
            values = line.split()
        
            theta_values.append(float(values[0]))
            chaos_values.append(float(values[1]))
            driven_values.append(float(values[2]))
            #damped_values.append(float(values[3]))
    data.close()
    fig = plt.figure(figsize=(10,10))
    axis = fig.add_subplot(111)
    zeros = np.zeros(len(theta_values))
    axis.plot(theta_values, chaos_values, color='black')
    axis.set_title("Maximum Lyapunov Exponents")
    axis.plot(theta_values, driven_values, color='blue')
    #axis.plot(theta_values, damped_values, color='orange')
    axis.set_xlabel(r"$\omega_D$ (Hz)")
    axis.set_xlim((0.1, 2))
    axis.grid()
    axis.set_ylabel(r"$\lambda_{\Theta}$", rotation=0)
    axis.legend(["Chaotic Model", "Damped Driven Model"])
    axis.plot(theta_values, zeros, linestyle='dashed', color='red')
    
    extent = axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Lyapunov_Exponents_Theta_drive_frequency.png", bbox_inches=extent.expanded(1.4, 1.4))

def compare_sim():
    filename = "Torsion Chaotic Data.xlsx"
    df = pd.read_excel(filename)
    lim = 1000
    angles = np.array(df["Angle (rad)"])[:lim]
    LE = nolds.lyap_r(angles, min_tsep = 10, lag = None)
    print(LE)
compare_sim()