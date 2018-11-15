import random
import csv
import numpy as np
import matplotlib.pyplot as plt

#  systemintervals = [1,0] 1 open, 0 do not open.

# List time intervals (15 Mins Interval in this case study) and  24 hour (1 day).
num_days = 7;  # 8 weeks
interval = 15;
dayWeek = 0;
num_timeIntervals = int(1 * (24 * (60 / interval)));  # 1 day!!

# Decision of the user. 1 if want open, 0 if not.

# Create random data (Hour,Minute,Sensor Data)

f = open('BedRoom.csv', "w")
k = csv.reader(f)

w = csv.writer(f)
w.writerow(('id', 'day', 'hour', 'minute', 'sensor_shutter', 'sensor_light', 'dayWeek'))

day = 1



for i in range(num_days):

    minute = 0
    hour = 0


    def openTime(dayWeek):  # [7:30] h

        #IF WeeKEnd
        if dayWeek == 5 or dayWeek ==  6:
            mean, stdev = 600, 30;
            s = np.random.normal(mean, stdev, 1);



        else:
            # Media 450 (7:30), desvio padrao de 15 minutos. 68,26% probabilidade de [7:15 - 8:30]
            mean, stdev = 450, 15;
            s = np.random.normal(mean, stdev, 28);


        # Create the bins and histogram
         #count, bins, ignored = plt.hist(s, 28, normed=True)

        # Plot the distribution curve
        #plt.plot(bins, 1 / (stdev * np.sqrt(2 * np.pi)) *
         #       np.exp(- (bins - mean) ** 2 / (2 * stdev ** 2)), linewidth=3, color='y')
        #plt.show()

        return s;





    def closeTime():  # [18 - 20] h
        # Media 1140 (19:00), desvio padrao de 15 minutos.   68.26% probabilidade de [18:15 - 19:45]
        mean, stdev = 1140, 15;
        s = np.random.normal(mean, stdev, 1);

        # Create the bins and histogram
        # count, bins, ignored = plt.hist(s, 20, normed=True)

        # Plot the distribution curve
        # plt.plot(bins, 1 / (stdev * np.sqrt(2 * np.pi)) *
        #        np.exp(- (bins - mean) ** 2 / (2 * stdev ** 2)), linewidth=3, color='y')
        # plt.show()

        return s


    def onTime_light():  # [18:30 - 21:25] h
        mean, stdev = 1185, 30;
        s = np.random.normal(mean, stdev, 1);

        # Create the bins and histogram
        #count, bins, ignored = plt.hist(s, 20, normed=True)

        #Plot the distribution curve
        #plt.plot(bins, 1 / (stdev * np.sqrt(2 * np.pi)) *
         #       np.exp(- (bins - mean) ** 2 / (2 * stdev ** 2)), linewidth=3, color='y')
         #plt.show()

        return s


    def offTime_light():  # [22 - 23:59] h
        mean, stdev = 1379.5, 15;
        s = np.random.normal(mean, stdev, 1);

        # Create the bins and histogram
        # count, bins, ignored = plt.hist(s, 20, normed=True)

        # Plot the distribution curve
        # plt.plot(bins, 1 / (stdev * np.sqrt(2 * np.pi)) *
        #        np.exp(- (bins - mean) ** 2 / (2 * stdev ** 2)), linewidth=3, color='y')
        # plt.show()

        return s


    for i in range(num_timeIntervals):

        def choice(hour, minute,dayWeek):
            time = hour * 60 + minute;
            open = openTime(dayWeek)
            close = closeTime()
            if open[0] <= time < close:
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


        w.writerow((i, day, hour, minute, choice(hour, minute,dayWeek), light(hour, minute), dayWeek))  # 0 - numIntervals-1
        minute = minute + interval;
        if minute == 60: hour, minute = hour + 1, 0
        if hour == 24: day, hour = day + 1, 0

    dayWeek = dayWeek + 1
    if dayWeek == 7: dayWeek = 0





f.close();
