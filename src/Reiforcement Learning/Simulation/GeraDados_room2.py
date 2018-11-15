import random
import csv
import numpy as np
import matplotlib.pyplot as plt

#  systemintervals = [1,0] 1 open, 0 do not open.

# List time intervals (15 Mins Interval in this case study) and  24 hour (1 day).
num_days = 56; #8 Weeks
interval = 15;
dayWeek = 0;
num_timeIntervals = int(1 * (24 * (60 / interval))); #1 day!!


# Decision of the user. 1 if want open, 0 if not.

# Create random data (Hour,Minute,Sensor Data)

f = open('BedRoom.csv', "a")
k = csv.reader(f)

w = csv.writer(f)
#w.writerow(('id', 'day', 'hour', 'minute', 'sensor_shutter','sensor_light','dayWeek'))

day=8
e=0.05

for i in range(num_days):


    minute = 0
    hour = 0


    def openTime():  # [15 - 16:30] h
        # desvio padrao de 15 minutos. 68,26% probabilidade de [7:15 - 8:30]
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

f.close();