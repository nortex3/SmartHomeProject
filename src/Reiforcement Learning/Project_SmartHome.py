# !/usr/bin/python3
from tkinter import *

import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import matplotlib
import random


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show

import os
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import keras
from sklearn.model_selection import train_test_split
import time
import datetime
from keras.models import Sequential
from keras.layers.core import Dense
import json
from pprint import pprint

import math

master = Tk()

# WITH THE GUI; WE CAN MAKE IT DINAMIC.

# List time intervals (15 Mins Interval in this case study) and  24 hour (1 day).
numberDay = 0
num_days = 0;
interval = 15;
intervalFixed = 15;
num_timeIntervals = int(1 * (24 * (60 / interval)));  # 1 day!!
numberIterations = 0
iteration = 0
initialIteration = 0
initialNumberDay = 0
name = None
rooms = []
actions = ["Shutter", "Lights"]
dataUsed = ["Historical Data", "Sensor"]
var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var4 = IntVar()
xTimes = IntVar()

# REWARDS FROM HISTORICAL DATA (INITIALIZE)

# N DIMENSIONAL MATRIX, [intervals,daysWeek] , being N number of rooms

reward_pa_open_shutter = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)
reward_pa_close_shutter = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)
reward_pa_light_on = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)
reward_pa_light_off = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)



# WEIGHT OF LUMINOSITY PAST DAYS
k = 10





 #  systemintervals = [1,0] 1 open, 0 do not open.



# Decision of the user. 1 if want open, 0 if not.

# Create random data (Hour,Minute,Sensor Data)




def geraDados(x):
    # List time intervals (15 Mins Interval in this case study) and  24 hour (1 day).
    num_days = 1;  # 8 Weeks
    interval = 15;
    dayWeek = 0;
    num_timeIntervals = int(1 * (24 * (60 / interval)));  # 1 day!!

    day = x
    e = 0.1

    f = open('BedRoom.csv', "a")
    k = csv.reader(f)

    w = csv.writer(f)
    #w.writerow(('id', 'day', 'hour', 'minute', 'sensor_shutter', 'sensor_light', 'dayWeek'))

    for i in range(num_days):
        minute = 0
        hour = 0

        def openTime():  # [15 - 16:30] h
            # desvio padrao de 15 minutos. 68,26% probabilidade de [15 - 16]
            mean, stdev = 945, 15;

            s = np.random.normal(mean, stdev, 1);

            # Create the bins and histogram
            # count, bins, ignored = plt.hist(s, 20, normed=True)

            # Plot the distribution curve
            # plt.plot(bins, 1 / (stdev * np.sqrt(2 * np.pi)) *
            #       np.exp(- (bins - mean) ** 2 / (2 * stdev ** 2)), linewidth=3, color='y')
            # plt.show()

            return s;

        def closeTime():  # [20 - 22] h
            # Media 1260, desvio padrao de 15 minutos.   68.26% probabilidade de [20:15 - 21:45]
            mean, stdev = 1260, 15;

            s = np.random.normal(mean, stdev, 1);

            return s;

        def onTime_light():  # [5:00 - 6:00] h
            # desvio padrao de 15 minutos. 68,26% probabilidade de [5:15 - 5:45]
            mean, stdev = 330, 15;

            s = np.random.normal(mean, stdev, 1);

            return s;

        def offTime_light():  # [6:30 - 7:30] h
            # Media 420, desvio padrao de 15 minutos.   68.26% probabilidade de [6:45 - 7:15]
            mean, stdev = 420, 15;

            s = np.random.normal(mean, stdev, 1);

            return s;

        for i in range(num_timeIntervals):

            def choice(hour, minute):
                time = hour * 60 + minute;
                open = openTime()
                close = closeTime()
                if open <= time < close:
                    return 1;
                else:
                    return 0;

            def light(hour, minute):
                time = hour * 60 + minute;
                on = onTime_light()
                off = offTime_light()
                if on <= time < off:
                    return 1;
                else:
                    return 0

            if np.random.rand(1) < e:
                w.writerow((i, day, hour, minute, random.randint(0, 1), random.randint(0, 1),
                            dayWeek))  # 0 - numIntervals-1
            else:
                w.writerow((i, day, hour, minute, choice(hour, minute), light(hour, minute),dayWeek))  # 0 - numIntervals-1

            minute = minute + interval;
            if minute == 60: hour, minute = hour + 1, 0
            if hour == 24: day, hour = day + 1, 0

        dayWeek = dayWeek + 1
        if dayWeek == 7: dayWeek = 0




def var_states():
    print(
        "Shutter: %d,\nLight: %d,\nHistorical Data: %d,\nSensor: %d" % (var1.get(), var2.get(), var3.get(), var4.get()))


dayWeek = {'Monday': 0, 'Tuesday': 1, 'Weednsday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}


# print("You entered " + str(aux1) + " + " + str(aux2));

def readCSV(room,n):
    global num_days
    print("!!!!!!!!!!!!!!!!!!!!! " + str(num_days) + "!!!!!!!!!!!!!!!!!!!!!!")
    extension = ".csv"
    roomVariable = room + extension
    df = pd.read_csv(roomVariable, sep=',', header=None)
    df2 = df.drop(df.index[0])
    df3 = df2.head((n+num_days)*96)
    return df3.values


# ---------------------------------------REWARDS FUNCTIONS---------------------------------------------------------#


# DO THE DISTANCE FROM RIGHT INTERVAL FUNCTION FOR PAST REWARD

# RewardPA = log(10,day) / dist()

# DIST(interval) = MIN (distEsq(),distDir())


# REWARD FUNCTION PAST VALUES FOR OPEN SHUTTERS

def distEsqOpen(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval > day * num_timeIntervals:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 0 and (int(userHistory) == 1 and int(systemSensor) == 1):
            find = 1
        interval = interval - 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 3
            return times
        else:
            # NEGATIVE WHEN MORE THAN 3
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distDirOpen(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval < day * num_timeIntervals + 96:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 0 and (int(userHistory) == 1 and int(systemSensor) == 1):
            find = 1
        interval = interval + 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 5
            return times
        else:
            # NEGATIVE WHEN MORE THAN 5
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distOpen(df, interval, column, day):
    distEsqOpen_aux = distEsqOpen(df, interval, column, day)
    distDirOpen_aux = distDirOpen(df, interval, column, day)
    if abs(distEsqOpen_aux) < abs(distDirOpen_aux):
        if distEsqOpen_aux > 0:
            return distEsqOpen_aux
        else:
            return 10 / distEsqOpen_aux
    else:
        if distDirOpen_aux > 0:
            return distDirOpen_aux
        else:
            return 10 / distDirOpen_aux


def pa_rewardFunctionOpenShutter(interval, df, dayWeek, z):
    column = 4
    day = dayWeek
    global num_days

    # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR!!!
    print("____________" + str(num_days) +"___________")

    while day < num_days:
        inter = interval + (day * num_timeIntervals)
        reward_pa_open_shutter[z, interval, dayWeek] = reward_pa_open_shutter[z, interval, dayWeek] + (
                (math.log(day + 1, 10)) / (distOpen(df, inter, column, day)))
        day = day + 7
    return reward_pa_open_shutter[z, interval, dayWeek]



# REWARD FUNCTION PAST VALUES FOR CLOSE SHUTTERS

def distEsqClose(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval > day * num_timeIntervals:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 1 and (int(userHistory) == 0 and int(systemSensor) == 0):
            find = 1
        interval = interval - 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 3
            return times
        else:
            # NEGATIVE WHEN MORE THAN 3
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distDirClose(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval < day * num_timeIntervals + 96:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 1 and (int(userHistory) == 0 and int(systemSensor) == 0):
            find = 1
        interval = interval + 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 5
            return times
        else:
            # NEGATIVE WHEN MORE THAN 5
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distClose(df, interval, column, day):
    distEsqClose_aux = distEsqClose(df, interval, column, day)
    distDirClose_aux = distDirClose(df, interval, column, day)
    if abs(distEsqClose_aux) < abs(distDirClose_aux):
        if distEsqClose_aux > 0:
            return distEsqClose_aux
        else:
            return 10 / distEsqClose_aux
    else:
        if distDirClose_aux > 0:
            return distDirClose_aux
        else:
            return 10 / distDirClose_aux


def pa_rewardFunctionCloseShutter(interval, df, dayWeek, z):
    column = 4
    day = dayWeek
    global num_days
    # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR!!!

    while day < num_days:
        # return a positive reward.
        inter = interval + (day * num_timeIntervals)
        reward_pa_close_shutter[z, interval, dayWeek] = reward_pa_close_shutter[z, interval, dayWeek] + (
                (math.log(day + 1, 10)) / (distClose(df, inter, column, day)))
        day = day + 7
    return reward_pa_close_shutter[z, interval, dayWeek]



# REWARD FUNCTION PAST VALUES FOR LIGHTS ON


def distEsqLightOn(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval > day * num_timeIntervals:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 0 and (int(userHistory) == 1 and int(systemSensor) == 1):
            find = 1
        interval = interval - 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 3
            return times
        else:
            # NEGATIVE WHEN MORE THAN 3
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distDirLightOn(df, interval, col, day):
    times = 0
    column = col
    find = 0

    while find == 0 and interval < day * num_timeIntervals + 96:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 0 and (int(userHistory) == 1 and int(systemSensor) == 1):
            find = 1
        interval = interval + 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 3
            return times
        else:
            # NEGATIVE WHEN MORE THAN 3
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distLightOn(df, interval, column, day):
    distEsqLightOn_aux = distEsqLightOn(df, interval, column, day)
    distDirLightOn_aux = distDirLightOn(df, interval, column, day)
    if abs(distEsqLightOn_aux) < abs(distDirLightOn_aux):
        if distEsqLightOn_aux > 0:
            return distEsqLightOn_aux
        else:
            return 10 / distEsqLightOn_aux
    else:
        if distDirLightOn_aux > 0:
            return distDirLightOn_aux
        else:
            return 10 / distDirLightOn_aux


def pa_rewardFunctionLightOn(interval, df, dayWeek, z):
    column = 5
    day = dayWeek
    global num_days

    # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR!!!

    while day < num_days:
        inter = interval + (day * num_timeIntervals)
        reward_pa_light_on[z, interval, dayWeek] = reward_pa_light_on[z, interval, dayWeek] + (
                    (math.log(day + 1, 10)) / (distLightOn(df, inter, column, day)))
        day = day + 7
    return reward_pa_light_on[z, interval, dayWeek]



# REWARD FUNCTION PAST VALUES FOR CLOSE SHUTTERS

def distEsqLightOff(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval > day * num_timeIntervals:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 1 and (int(userHistory) == 0 and int(systemSensor) == 0):
            find = 1
        interval = interval - 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 3
            return times
        else:
            # NEGATIVE WHEN MORE THAN 3
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distDirLightOff(df, interval, col, day):
    times = 0
    column = col
    find = 0
    while find == 0 and interval < day * num_timeIntervals + 96:
        systemSensor = df[interval, column]
        userHistory = df[interval, column]
        userHistoryBefore = df[(interval) - 1, column]
        if int(userHistoryBefore) == 1 and (int(userHistory) == 0 and int(systemSensor) == 0):
            find = 1
        interval = interval + 1
        times = times + 1
    if find == 1:
        if times <= 5:
            # POSITIVE WHEN LESS THAN 5
            return times
        else:
            # NEGATIVE WHEN MORE THAN 5
            return -times
    else:
        # WHEN OUT OF BOUNDS.
        return 1000


def distLightOff(df, interval, column, day):
    distEsqLightOff_aux = distEsqLightOff(df, interval, column, day)
    distDirLightOff_aux = distDirLightOff(df, interval, column, day)
    if abs(distEsqLightOff_aux) < abs(distDirLightOff_aux):
        if distEsqLightOff_aux > 0:
            return distEsqLightOff_aux
        else:
            return 10 / distEsqLightOff_aux
    else:
        if distDirLightOff_aux > 0:
            return distDirLightOff_aux
        else:
            return 10 / distDirLightOff_aux


def pa_rewardFunctionLightOff(interval, df, dayWeek, z):
    column = 5
    day = dayWeek
    global num_days
    # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR!!!

    while day < num_days:
        # return a positive reward.
        inter = interval + (day * num_timeIntervals)
        reward_pa_light_off[z, interval, dayWeek] = reward_pa_light_off[z, interval, dayWeek] + (
                (math.log(day + 1, 10)) / (distLightOff(df, inter, column, day)))
        day = day + 7
    return reward_pa_light_off[z, interval, dayWeek]



# REWARD FUNCTION TOTAL FOR OPEN SHUTTERS

def rewardFunctionOpenShutter(interval, df, dayWeek, z,luminosity_sensor):
    # LUMINOSITY PREDICTION FOR TODAY, K IS THE WEIGHT
    return (pa_rewardFunctionOpenShutter(interval, df, dayWeek, z) * var3.get() + (
            luminosity_sensor[interval] * k) * var4.get()) / (total_episodes / 10)


# REWARD FUNCTION TOTAL FOR OPEN SHUTTERS

def rewardFunctionCloseShutter(interval, df, dayWeek, z,luminosity_sensor):
    # LUMINOSITY PREDICTION FOR TODAY, K IS THE WEIGHT
    return (pa_rewardFunctionCloseShutter(interval, df, dayWeek, z) * var3.get() - (
            luminosity_sensor[interval] * k) * var4.get()) / (total_episodes / 10)


# REWARD FUNCTION TOTAL FOR LIGHTS ON

def rewardFunctionLightOn(interval, df, dayWeek, z,luminosity_sensor):
    # LUMINOSITY PREDICTION FOR TODAY , K IS THE WEIGHT
    return (pa_rewardFunctionLightOn(interval, df, dayWeek, z) * var3.get() - (
            luminosity_sensor[interval] * k) * var4.get()) / (total_episodes / 10)


# REWARD FUNCTION TOTAL FOR LIGHTS OFF

def rewardFunctionLightOff(interval, df, dayWeek, z,luminosity_sensor):
    # LUMINOSITY PREDICTION FOR TODAY , K IS THE WEIGHT
    return (pa_rewardFunctionLightOff(interval, df, dayWeek, z) * var3.get() + (
            luminosity_sensor[interval] * k) * var4.get()) / (total_episodes / 10)


# ---------------------------------------REWARDS FUNCTIONS---------------------------------------------------------#


# PREDICT THE BEST interval  ( THE AGENT )
tf.reset_default_graph()

# These two lines established the feed-forward part of the network. This does the actual choosing.

weights = tf.Variable(tf.ones([num_timeIntervals]))  # OUTPUT
chosen_interval = tf.argmax(weights, 0)

# The next six lines establish the training proceedure. We feed the reward and chosen interval into the network
# to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
interval_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, interval_holder, [1])

# Loss = -log(n)*A
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# Training the Agent

total_episodes = 5000  # Set  .

# DEFINE THE TOTAL REWARD VARIABLES. MATRIX WITH NUMBER OF INTERVALS AND DAY OF THE WEEK.

# Shutter
total_reward = np.zeros((2, num_timeIntervals, 7),
                        dtype=np.float64)  # Set scoreboard for intervals to 0.  7 days each week
total_reward_Close = np.zeros((2, num_timeIntervals, 7),
                              dtype=np.float64)  # Set scoreboard for intervals to 0.  7 days each week
PA_reward = np.zeros((2, num_timeIntervals, 7),
                     dtype=np.float64)  # Set scoreboard for intervals to 0.  7 days each week
PA_reward_Close = np.zeros((2, num_timeIntervals, 7),
                           dtype=np.float64)  # Set scoreboard for intervals to 0.  7 days each week

# Lights
total_reward_lightOn = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)  # Set scoreboard for intervals to 0.
total_reward_lightOff = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)  # Set scoreboard for intervals to 0.
PA_reward_lightOn = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)  # Set scoreboard for intervals to 0.
PA_reward_lightOff = np.zeros((2, num_timeIntervals, 7), dtype=np.float64)  # Set scoreboard for intervals to 0.

# EXPLOITATION VS EXPLORATION!

# HIGH NUMBER , BECAUSE THERE ARE 96 Intervals.

e = 0.75  # Set the chance of taking a random interval.

list_intervals = []
list_intervalsClose = []

list_intervals_lightOff = []
list_intervals_lightOn = []

init = tf.global_variables_initializer()


def main():
    def reinforcement_learning():
        global iteration, initialIteration, initialNumberDay, numberDay,num_days
        global numberIterations



        def algorithm(room, z):
            global numberDay
            global iteration
            global xTimes
            global num_days
            global numberIterations


            n = 0
            while (n < xTimes.get()):
                # In Windows


                # FOR NOW; ITS RANDOM
                luminosity_sensor = np.random.rand(num_timeIntervals)
                # ---------------SHUTTERS----------------------------------------------------------------------------------------

                # Launch the tensorflow graph ((BEST TIME TO OPEN))
                if (var1.get() == 1):
                    with tf.Session() as sess:
                        print("-------------------- OPEN --------------------------")
                        sess.run(init)
                        df = readCSV(room,num_days+1)
                        j = 0

                        while j < total_episodes:

                            # Choose either a random interval or one from our network.
                            if np.random.rand(1) < e:
                                interval = np.random.randint(1, num_timeIntervals)  # Choose one interval
                            else:
                                interval = sess.run(chosen_interval)  # Choose the best to interval to do the interval


                            reward = rewardFunctionOpenShutter(interval, df,
                                                               iteration,
                                                               z,luminosity_sensor)  # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR!!!

                            # Update the network.
                            _, resp, ww = sess.run([update, responsible_weight, weights],
                                                   feed_dict={reward_holder: [reward], interval_holder: [interval]})

                            # Update our running tally of scores.
                            total_reward[z, interval, iteration] += reward
                            list_intervals.append(interval)

                            if j % 5000 == 0:
                                print("Running reward for the " + str(num_timeIntervals) + " time intervals: " + str(
                                    total_reward[z, :, iteration]))
                            j += 1

                    print("-------------------- OPEN --------------------------")

                    # CALCULATE TOTAL REWARD FOR THIS ACTION (PAST ACTIVITIES + API + SENSORS)

                    # WEIGHT OF LUMINOSITY
                    # k = 0;
                    # for i in range(0, num_timeIntervals):
                    #    total_reward[i] = PA_reward[i] + (k * luminosity_sensor[i])

                time.sleep(1)

                # Launch the tensorflow graph ((BEST TIME TO CLOSE))
                if (var1.get() == 1):
                    with tf.Session() as sess:
                        print("-------------------- CLOSE --------------------------")
                        sess.run(init)
                        df = readCSV(room,num_days+1)
                        j = 0

                        while j < total_episodes:

                            # Choose either a random interval or one from our network.
                            if np.random.rand(1) < e:
                                interval = np.random.randint(1, num_timeIntervals)  # Choose one interval (FOR OPEN)
                            # interval = np.random.randint(0,num_timeIntervals-1) #FOR CLOSE
                            else:
                                interval = sess.run(chosen_interval)  # Choose the best to interval to do the interval

                            reward_Close = rewardFunctionCloseShutter(
                                interval, df,
                                iteration,
                                z,
                                luminosity_sensor)  # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR!!!

                            # Update the network.
                            _, resp, wwClose = sess.run([update, responsible_weight, weights],
                                                        feed_dict={reward_holder: [reward_Close],
                                                                   interval_holder: [interval]})

                            # Update our running tally of scores.
                            total_reward_Close[z, interval, iteration] += reward_Close
                            list_intervalsClose.append(interval)

                            if j % 5000 == 0:
                                print("Running reward for the " + str(num_timeIntervals) + " time intervals: " + str(
                                    total_reward_Close[z, :, iteration]))
                            j += 1

                    print("-------------------- CLOSE --------------------------")

                # CALCULATE TOTAL REWARD FOR THIS ACTION (PAST ACTIVITIES + API + SENSORS)

                # WEIGHT OF Sensor
                # k = 0;

                # for i in range(0, num_timeIntervals):
                #    total_reward_Close[i] = PA_reward_Close[i] - (k * luminosity_sensor[i])

                # ---------------SHUTTERS----------------------------------------------------------------------------------------
                time.sleep(1)

                # ---------------LIGHTS------------------------------------------------------------------------------------------

                # LIGHT ON!

                if (var2.get() == 1):
                    with tf.Session() as sess:
                        print("-------------------- LIGHT ON --------------------------")
                        sess.run(init)
                        df = readCSV(room,num_days+1)
                        j = 0
                        while j < total_episodes:


                            # Choose either a random interval or one from our network.
                            if np.random.rand(1) < e:
                                interval = np.random.randint(1, num_timeIntervals)  # Choose one interval
                            else:
                                interval = sess.run(chosen_interval)  # Choose the best to interval to do the interval

                            reward_On = rewardFunctionLightOn(interval, df,
                                                              iteration,
                                                              z,
                                                              luminosity_sensor)  # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR LIGHT!!!
                            # Update the network.
                            _, resp, wwOn = sess.run([update, responsible_weight, weights],
                                                     feed_dict={reward_holder: [reward_On],
                                                                interval_holder: [interval]})

                            # Update our running tally of scores.
                            total_reward_lightOn[z, interval, iteration] += reward_On
                            list_intervals_lightOn.append(interval)

                            if j % 5000 == 0:
                                print("Running reward for the " + str(num_timeIntervals) + " time intervals: " + str(
                                    total_reward_lightOn[z, :, iteration]))
                            j += 1
                        print("-------------------- Light ON --------------------------")

                time.sleep(1)

                # LIGHT OFF

                if (var2.get() == 1):
                    with tf.Session() as sess:
                        print("-------------------- LIGHT OFF --------------------------")
                        sess.run(init)
                        df = readCSV(room,num_days+1)
                        j = 0
                        while j < total_episodes:

                            # Choose either a random interval or one from our network.
                            if np.random.rand(1) < e:
                                interval = np.random.randint(1, num_timeIntervals)  # Choose one interval
                            else:
                                interval = sess.run(chosen_interval)  # Choose the best to interval to do the interval

                            reward_Off = rewardFunctionLightOff(interval, df,
                                                                iteration,
                                                                z,
                                                                luminosity_sensor)  # Get our reward from picking one of the timeIntervals. COLUMN 4 is SENSOR LIGHT!!!
                            # Update the network.
                            _, resp, wwOff = sess.run([update, responsible_weight, weights],
                                                      feed_dict={reward_holder: [reward_Off],
                                                                 interval_holder: [interval]})

                            # Update our running tally of scores.
                            total_reward_lightOff[z, interval, iteration] += reward_Off
                            list_intervals_lightOff.append(interval)

                            if j % 5000 == 0:
                                print("Running reward for the " + str(num_timeIntervals) + " time intervals: " + str(
                                    total_reward_lightOff[z, :, iteration]))
                            j += 1
                        print("-------------------- LIGHT OFF --------------------------")

                # ---------------LIGHTS------------------------------------------------------------------------------------------

                pprint(total_reward[z, :, iteration])

                extension = ".csv"
                roomVariable = room + extension
                f = open(roomVariable, 'a')
                k = csv.reader(f)
                w = csv.writer(f)

                # CSV with intervals chosen

                f1 = open('ChoosenIntervals.csv', 'a')
                k1 = csv.reader(f1)
                w1 = csv.writer(f1)

                w1.writerow((numberDay, np.argmax(total_reward[z, :, iteration]),
                             np.argmax(total_reward_Close[z, :, iteration]),
                             np.argmax(total_reward_lightOn[z, :, iteration]),
                             np.argmax(total_reward_lightOff[z, :, iteration]), iteration))

                minute = 0
                hour = 0

                def openTime():  # Chosen by algorithm, Best interval to open Shutter
                    return np.argmax(total_reward[z, :, iteration])

                def closeTime():  # Chosen by algorithm, Best interval to Close Shutter
                    return np.argmax(total_reward_Close[z, :, iteration])

                def onTime_light():  # Chosen by algorithm, Best interval to Close Shutter
                    return np.argmax(total_reward_lightOn[z, :, iteration])

                def offTime_light():  # Chosen by algorithm, Best interval to Close Shutter
                    return np.argmax(total_reward_lightOff[z, :, iteration])

                for i in range(num_timeIntervals):

                    def choice(hour, minute):
                        time = hour * 60 + minute;
                        intervalAux = time / 15
                        open = openTime()
                        close = closeTime()
                        if open <= intervalAux < close:
                            return 1;
                        else:
                            return 0;

                    def light(hour, minute):
                        time = hour * 60 + minute;
                        intervalAux = time / 15
                        on = onTime_light()
                        off = offTime_light()
                        if on <= intervalAux < off:
                            return 1;
                        else:
                            return 0


                    minute = minute + intervalFixed;
                    if minute == 60: hour, minute = hour + 1, 0
                    if hour == 24: hour = 0

                numberIterations +=1
                iteration += 1
                if (iteration == 7):
                    iteration = 0
                numberDay += 1
                n = n + 1

                num_days += 1
                #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!"+str(num_days)+"!!!!!!!!!!!!!!!!!!!!!1")

            iterationResult = iteration - 1
            pprint(total_reward[z, :, iteration])


            if (var1.get() == 1):
                print("The agent thinks time Interval " + str(
                    np.argmax(total_reward[z, :, iterationResult])) + " || " + str('{:02d}:{:02d}'.format(*divmod(int(
                    np.argmax(total_reward[z, :, iterationResult])) * 15,
                                                                                                                  60))) + " is the best to open Shutters for day " + str(
                    list(dayWeek.keys())[iterationResult]) + " in room " + str(room))
                print("The agent thinks time Interval " + str(
                    np.argmax(total_reward_Close[z, :, iterationResult])) + " || " + str(
                    '{:02d}:{:02d}'.format(*divmod(int(
                        np.argmax(total_reward_Close[z, :, iterationResult])) * 15,
                                                   60))) + " is the best to Close Shutters " + str(
                    list(dayWeek.keys())[iterationResult]) + " in room " + str(room))
            if (var2.get() == 1):
                print(
                    "The agent thinks time Interval " + str(
                        np.argmax(total_reward_lightOn[z, :, iterationResult])) + " || " + str(
                        '{:02d}:{:02d}'.format(*divmod(int(
                            np.argmax(total_reward_lightOn[z, :, iterationResult])) * 15,
                                                       60))) + " is the best to Turn on the lights " + str(
                        list(dayWeek.keys())[iterationResult]) + " in room " + str(room))
                print("The agent thinks time Interval " + str(
                    np.argmax(total_reward_lightOff[z, :, iterationResult])) + " || " + str(
                    '{:02d}:{:02d}'.format(*divmod(int(
                        np.argmax(total_reward_lightOff[z, :, iterationResult])) * 15,
                                                   60))) + " is the best to Turn off the lights " + str(
                    list(dayWeek.keys())[iterationResult]) + " in room " + str(room))




            def print_intervals():
                plt.figure(1)
                plt.ion()
                num_bins = 48
                if (var1.get() == 1):
                    plt.hist(list_intervals, num_bins)
                    plt.hist(list_intervalsClose, num_bins)
                if (var2.get() == 1):
                    plt.hist(list_intervals_lightOn, num_bins)
                    plt.hist(list_intervals_lightOff, num_bins)
                plt.title('Interval Chosen ' + str(room))
                plt.ylabel('number of times ' + str(room))

                if (var1.get() == 1 and var2.get() == 0):
                    plt.legend(['Open Shutter', 'Close Shutter'], loc='upper left')
                elif (var1.get() == 0 and var2.get() == 1):
                    plt.legend(['Light ON', 'Light OFF'], loc='upper left')
                elif (var1.get() == 1 and var2.get() == 1):
                    plt.legend(['Open Shutter', 'Close Shutter', 'Light ON', 'Light OFF'], loc='upper left')

                plt.xlabel('Interval')
                plt.draw()
                plt.savefig('intervals' + str(room) + '.png')
                plt.pause(5)

            def print_rewards():
                plt.figure(2)
                plt.ion()
                if (var1.get() == 1):
                    plt.plot(total_reward[z, :, iterationResult])
                    plt.plot(total_reward_Close[z, :, iterationResult])
                if (var2.get() == 1):
                    plt.plot(total_reward_lightOn[z, :, iterationResult])
                    plt.plot(total_reward_lightOff[z, :, iterationResult])
                plt.title('Best Interval ' + str(room))
                plt.ylabel('reward ' + str(room))
                plt.xlabel('Intervals ' + str(room))

                if (var1.get() == 1 and var2.get() == 0):
                    plt.legend(['Open Shutter', 'Close Shutter'], loc='upper left')
                elif (var1.get() == 0 and var2.get() == 1):
                    plt.legend(['Light ON', 'Light OFF'], loc='upper left')
                elif (var1.get() == 1 and var2.get() == 1):
                    plt.legend(['Open Shutter', 'Close Shutter', 'Light ON', 'Light OFF'], loc='upper left')

                plt.draw()
                plt.savefig('rewards' + str(room) + '.png')
                plt.pause(5)

            print_intervals()
            print_rewards()

            # Agente Interface, Decide melhor action para determinado intervalo de tempo (comparando reward de cada agente)
            satisfied = 0
            while (satisfied== 0):

                var = input("Please enter an interval of the day (0-96):  ")
                var = int(var);
                print("You entered " + str('{:02d}:{:02d}'.format(*divmod(var * 15, 60))))
                plt.close("all")

                LightON = total_reward_lightOn[z, int(var), iterationResult]
                LightOff = total_reward_lightOff[z, int(var), iterationResult]
                ShutterON = total_reward[z, int(var), iterationResult]
                ShutterOff = total_reward_Close[z, int(var), iterationResult]

                if (var1.get() == 1 and var2.get() == 0):
                    Actions = {'Open Shutter': ShutterON, 'Close Shutter': ShutterOff}
                elif (var1.get() == 0 and var2.get() == 1):
                    Actions = {'ON Light': LightON, 'Off Light': LightOff}
                elif (var1.get() == 1 and var2.get() == 1):
                    Actions = {'Open Shutter': ShutterON, 'Close Shutter': ShutterOff, 'ON Light': LightON,
                           'Off Light': LightOff}

                best_action = max(Actions, key=Actions.get)

                print("The best action for Interval " + str(var) + " is " + str(best_action) + " in the day " + str(
                    list(dayWeek.keys())[iterationResult]))

                concorda2 = input("Do You Agree? (YES/NO) :  ")
                print("You entered " + str(concorda2))

                # FEEDBACK, ONLY AFFECTS PAST ACIVITIES (PA) REWARD
                print(best_action)
                time.sleep(3)
                if (str(concorda2) == 'YES'):
                    if (best_action == "ON Light"):
                        total_reward_lightOn[z, int(var), iterationResult] = total_reward_lightOn[
                                                                             z, int(var), iterationResult] + 10*numberIterations
                    elif (best_action == "Off Light"):
                        total_reward_lightOff[z, int(var), iterationResult] = total_reward_lightOff[
                                                                              z, int(var), iterationResult] + 10*numberIterations
                    elif (best_action == "Open Shutter"):
                        total_reward[z, int(var), iterationResult] = total_reward[z, int(var), iterationResult] + 10*numberIterations
                    elif (best_action == "Close Shutter"):
                        total_reward_Close[z, int(var), iterationResult] = total_reward_Close[
                                                                           z, int(var), iterationResult] + 10*numberIterations

                elif (str(concorda2) == 'NO'):
                    if (best_action == "ON Light"):
                        total_reward_lightOn[z, int(var), iterationResult] = total_reward_lightOn[
                                                                             z, int(var), iterationResult] - (
                                                                             5*numberIterations)
                    elif (best_action == "Off Light"):
                        total_reward_lightOff[z, int(var), iterationResult] = total_reward_lightOff[
                                                                              z, int(var), iterationResult] - (
                                                                              10*numberIterations)
                    elif (best_action == "Open Shutter"):
                        total_reward[z, int(var), iterationResult] = total_reward[z, int(var), iterationResult] - (10*numberIterations)
                    elif (best_action == "Close Shutter"):
                        total_reward_Close[z, int(var), iterationResult] = total_reward_Close[
                                                                           z, int(var), iterationResult] - 10*numberIterations
                varS = input("Conclude? 0/1 : ")
                varS = int(varS);
                if varS == 1 : satisfied = 1
                else: satisfied=0

        z = 0

        while (z < len(rooms)):
            iteration = initialIteration
            numberDay = initialNumberDay
            print(str(rooms[z]))
            print("Processing Next Room")
            time.sleep(3)
            algorithm(rooms[z], z)
            z = z + 1
        # MAIN CICLE
        initialIteration = iteration
        initialNumberDay = numberDay
        label_1['text'] = "DAY: " + str(initialNumberDay + 1) + ", " + str(list(dayWeek.keys())[initialIteration])

    master.after(total_episodes, reinforcement_learning())


def show_results():
    z = 0
    iterationResult = iteration - 1
    if (var1.get() == 1):
        ANS = ("The agent thinks time Interval " + str(np.argmax(total_reward[z, :, iterationResult])) + " || " + str(
            '{:02d}:{:02d}'.format(*divmod(int(
                np.argmax(total_reward[z, :, iterationResult])) * 15,
                                           60))) + " is the best to open Shutters for day " + str(
            list(dayWeek.keys())[iterationResult]) + " in room " + str(rooms) +
               "\n\nThe agent thinks time Interval " + str(
                    np.argmax(total_reward_Close[z, :, iterationResult])) + " || " + str(
                    '{:02d}:{:02d}'.format(*divmod(int(
                        np.argmax(total_reward_Close[z, :, iterationResult])) * 15,
                                                   60))) + " is the best to Close Shutters " + str(
                    list(dayWeek.keys())[iterationResult]) + " in room " + str(rooms) +

               # if (var2.get() == 1):

               "\n\nThe agent thinks time Interval " + str(
                    np.argmax(total_reward_lightOn[z, :, iterationResult])) + " || " + str(
                    '{:02d}:{:02d}'.format(*divmod(int(
                        np.argmax(total_reward_lightOn[z, :, iterationResult])) * 15,
                                                   60))) + " is the best to Turn on the lights " + str(
                    list(dayWeek.keys())[iterationResult]) + " in room " + str(rooms) +
               "\n\nThe agent thinks time Interval " + str(
                    np.argmax(total_reward_lightOff[z, :, iterationResult])) + " || " + str(
                    '{:02d}:{:02d}'.format(*divmod(int(
                        np.argmax(total_reward_lightOff[z, :, iterationResult])) * 15,
                                                   60))) + " is the best to Turn off the lights " + str(
                    list(dayWeek.keys())[iterationResult]) + " in room " + str(rooms))

    messagebox.showinfo(message=ANS)


# Etapa 1 - preparar o dataset

def Neural_Network():
    def get_stock_data(normalized=0, file_name=None):
        stocks = pd.read_csv("BedRoom.csv", header=0, usecols=[1, 2, 3, 4, 6],
                             delimiter=',')  # fica numa especie de tabela exactamente como estava no csv )
        df = pd.DataFrame(stocks)
        return df

    def load_stock_dataset():
        return get_stock_data(0, 'Room1.csv')

    # Visualizar os top registos da tabela
    def visualize():
        df = load_stock_dataset()
        print('### Antes do pre-processamento ###')
        print(df.head())  # mostra so os primeiros 5 registos

    class TimeHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, batch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, batch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)

    # utils para visulaizacao do historial de aprendizagem
    def print_history_accuracy(history):
        print(history.history.keys())
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def print_history_loss(history):
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def build_model2():
        model = Sequential()
        model.add(Dense(64, input_dim=4, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def print_model(model, fich):
        from keras.utils import plot_model
        plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

    def model_evaluate(model, input_attributes, output_attributes):
        print("###########inicio do evaluate###############################\n")
        scores = model.evaluate(input_attributes, output_attributes)
        print("\n metrica: %s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))

    def print_series_prediction(y_test, predic):
        diff = []
        racio = []
        for i in range(len(y_test)):  # para imprimir tabela de previsoes
            racio.append((y_test[i] / predic[i]) - 1)
            diff.append(abs(y_test[i] - predic[i]))
            print('valor: %f ---> Previsao: %f Diff: %f Racio: %f' % (y_test[i], predic[i], diff[i], racio[i]))
        plt.plot(y_test, color='blue', label='y_test')
        plt.plot(predic, color='red', label='prediction')  # este deu uma linha em branco
        plt.plot(diff, color='green', label='diff')
        plt.plot(racio, color='yellow', label='racio')
        plt.legend(loc='upper left')
        plt.show()

    def LSTM_utilizando_data():
        df = load_stock_dataset()
        print("df", df.shape)
        x = df.iloc[:, [0, 1, 2, 4]]
        y = df.iloc[:, 3]  # for SHUTTER
        # y = df.iloc[:, 4]  # for LIGHTS
        (X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.4)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        print("X_test", X_test.shape)
        print('y_test', y_test.shape)
        model = build_model2()
        time_callback = TimeHistory()
        history = model.fit(X_train, y_train, callbacks=[time_callback], batch_size=256, epochs=1000, verbose=1)
        times = time_callback.times
        print_history_loss(history)
        print_model(model, "lstm_model.png")
        trainScore = model.evaluate(X_train, y_train, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
        testScore = model.evaluate(X_test, y_test, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
        print(model.metrics_names)
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        p = model.predict(X_test)
        predic = np.squeeze(
            np.asarray(p))  # para transformar uma matriz de uma coluna e n linhas em um np array de n elementos
        predicRounded = np.rint(predic)  # para arrendondar para o inteiro mais proximo!
        print_series_prediction(y_test, predicRounded)
        print('Total Time of Execution: %.2f seconds' % (sum(times)))

    LSTM_utilizando_data()


# + ACTION SELECTION
# + DATA SELECTION
# + ROOM SELECTION
# + DAY OF THE WEEK + DAY Information
# OUTPUT.


# NEEDS : feedback interactionn + querys?


# ----------------------------------------- GUI; Interface with TKINTER--------------------------------------------

master.geometry("240x600+0+0")
master.title("Smart Home")
menubar = Menu(master)
master.config(menu=menubar)
Label(master, text="Load a profile").pack()


def onNew():
    pprint("CREATE NEW PROFILE")


def onOpen():
    # READ FILE
    global numberDay, initialNumberDay, iteration, initialIteration, name, rooms, reward_pa_open_shutter, reward_pa_close_shutter, reward_pa_light_on, reward_pa_light_off, dlg, num_days
    cwd = os.getcwd()

    ftypes = [('Json files', '*.json'), ('All files', '*')]
    dlg = filedialog.askopenfilename(initialdir=cwd, title="Select file", filetypes=ftypes)

    base = os.path.basename(dlg)
    file_name = os.path.splitext(base)[0]

    # LOAD USER PROFILES

    with open(dlg) as json_data:
        d_json = json.load(json_data)

    numberDay = d_json["day"]
    iteration = d_json["dayWeek"]
    name = d_json["name"]
    rooms = d_json["rooms"]
    reward_pa_open_shutter = np.array(d_json["reward_pa_open_shutter"])
    reward_pa_close_shutter = np.array(d_json["reward_pa_close_shutter"])
    reward_pa_light_on = np.array(d_json["reward_pa_light_on"])
    reward_pa_light_off = np.array(d_json["reward_pa_light_off"])
    pprint(name)
    pprint(reward_pa_open_shutter)

    num_days = numberDay - 1

    initialIteration = iteration
    initialNumberDay = numberDay

    Label(master, text="Choose the following actions to automate:").pack()

    Checkbutton(master, text=str(actions[0]), variable=var1).pack()
    Checkbutton(master, text=str(actions[1]), variable=var2).pack()

    Label(master, text="Choose the following Room:").pack()
    count = 0
    for item in rooms:
        chk = Checkbutton(master, text=rooms[count], state=DISABLED)
        count += 1
        chk.pack()

    Label(master, text="Data used:").pack()

    Checkbutton(master, text=str(dataUsed[0]), variable=var3).pack()
    Checkbutton(master, text=str(dataUsed[1]), variable=var4).pack()

    def sel():
        selection = "Value = " + str(xTimes.get())
        label.config(text=selection)

    scale = Scale(master, variable=xTimes, orient="horizontal")
    scale.pack()

    button = Button(master, text="Get Scale Value", command=sel)
    button.pack()

    Button(master, text='Quit', command=master.quit).pack()
    Button(master, text='Show', command=var_states).pack()
    Button(master, text='Reinforcement Learning', command=main).pack()
    Button(master, text='Output', command=show_results).pack()
    Button(master, text='NeuralNetwork', command=Neural_Network).pack()
    Button(master, text='SAVE', command=saveProfile).pack()

    label_1['text'] = "DAY: " + str(numberDay + 1) + ", " + str(list(dayWeek.keys())[iteration])
    label_0['text'] = "WELCOME " + str(name)


label = Label(master)
label.pack()
label_1 = Label(master, text="DAY: " + str(numberDay + 1) + ", " + str(list(dayWeek.keys())[iteration]), bg="red")
label_1.pack()
label_0 = Label(master, text="WELCOME " + str(name))
label_0.pack()

fileMenu = Menu(menubar)
fileMenu.add_command(label="Open", command=onOpen)
fileMenu.add_command(label="New", command=onNew)
menubar.add_cascade(label="File", menu=fileMenu)


# SAVE USER PROFILE (OVERWRITE) Example!!

def saveProfile():
    global numberDay, iteration, name, rooms, reward_pa_open_shutter, reward_pa_close_shutter, reward_pa_light_on, reward_pa_light_off, dlg

    a1 = reward_pa_open_shutter.tolist()
    a2 = reward_pa_close_shutter.tolist()
    a3 = reward_pa_light_on.tolist()
    a4 = reward_pa_light_off.tolist()

    data = {
        "name": name,
        "rooms": rooms,
        "day": numberDay,
        "dayWeek": iteration,
        "reward_pa_open_shutter": a1,
        "reward_pa_close_shutter": a2,
        "reward_pa_light_on": a3,
        "reward_pa_light_off": a4
    }

    with open(dlg, "w") as write_file:
        json.dump(data, write_file, sort_keys=True, indent=4)


# Button(master, text='RESET', command=reset()).grid(row=9, sticky=W, pady=4)


mainloop()
