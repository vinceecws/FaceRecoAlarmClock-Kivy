from datetime import datetime, date, time, timedelta
import pickle

class AlarmManager():

    def __init__(self, alarms_dir=None):
        if alarms_dir: 
            self.loadAlarms(alarms_dir)
        else:
            self.alarms = []

    def __len__(self):
        return len(self.alarms)

    def __str__(self):
        return [str(alarm) for alarm in self.alarms]

    def getAlarm(self, idx):
        return self.alarms[idx]

    def checkAndTriggerAlarms(self): #Checked every minute
        active = filter(lambda alarm: alarm.isActive(), self.alarms)
        triggeredAlarms = []
        triggered = False
        for alarm in active:
            if alarm.dateAndTime == datetime.now().replace(second=0, microsecond=0):
                triggeredAlarms.append(alarm)
                triggered = True

        return triggered, triggeredAlarms

    def addAlarm(self, hour, minute, notation, label=None): #Alarm inactive by default, to trigger checking in activate()
        hour = int(hour)
        minute = int(minute)

        assert hour in range(1,13) and minute in range(0, 60) and notation in ['AM', 'PM']
        if notation == 'PM':
            if hour != 12:
                hour += 12
        elif notation == 'AM':
            if hour == 12:
                hour = 0

        if label is None:
            label = ''

        alarm = Alarm(-1, hour, minute, active=True, label=label)
        self.alarms.append(alarm)
        self.sortAlarms()

        return alarm.getIndex()

    def editAlarm(self, idx, hour, minute, notation, label):
        hour = int(hour)
        minute = int(minute)

        assert hour in range(1,13) and minute in range(0, 60) and notation in ['AM', 'PM']
        if notation == 'PM':
            if hour != 12:
                hour += 12
        elif notation == 'AM':
            if hour == 12:
                hour = 0

        if label is None:
            label = ''

        self.alarms[idx].setTime(hour, minute)
        self.alarms[idx].setLabel(label)
        self.sortAlarms()

        return idx

    def removeAlarm(self, idx):
        del self.alarms[idx]
        self.sortAlarms()

    def activateAlarm(self, idx):
        self.alarms[idx].activate()

    def deactivateAlarm(self, idx):
        self.alarms[idx].deactivate()

    def sortAlarms(self):
        self.alarms.sort(key=lambda alarm: alarm.dateAndTime.time())
        for ind, alarm in enumerate(self.alarms):
            alarm.setIndex(ind)

    def loadAlarms(self, alarms_dir):
        with open(alarms_dir, "rb") as file: 
            self.alarms = pickle.load(file)

    def saveAlarms(self, alarms_dir):
        with open(alarms_dir, "wb") as file:
            pickle.dump(self.alarms, file)

class Alarm():
    def __init__(self, index, hour, minute, active=True, label=None):
        self.index = index
        self.dateAndTime = datetime.now()
        self.setTime(hour=hour, minute=minute)
        self.label = label
        if active:
            self.activate()

    def __str__(self):
        mode = 'ON' if self.active else 'OFF'
        time, notation = self.get12()
        return f'Index: {self.index}, Time: {time}{notation}, Label: {self.label}, Mode: {mode}'

    def get12(self):
        time = self.dateAndTime.strftime("%I:%M")
        notation = self.dateAndTime.strftime("%p")

        return time, notation

    def get24(self):
        time = self.dateAndTime.strftime("%H%M")
        notation = 'HRS'
        return time, notation

    def setIndex(self, index):
        self.index = index

    def setTime(self, hour, minute, second=0, microsecond=0):
        self.dateAndTime = self.dateAndTime.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)

    def setDate(self, year, month, day):
        self.dateAndTime = self.dateAndTime.replace(year=year, month=month, day=day)

    def setActive(self, active):
        self.active = active

    def setLabel(self, label):
        self.label = label

    def getIndex(self):
        return self.index

    def getHour(self):
        return self.dateAndTime.strftime("%I")

    def getMinute(self):
        return self.dateAndTime.strftime("%M")

    def getNotation(self):
        return self.dateAndTime.strftime("%p")

    def getLabel(self):
        return self.label

    def isActive(self):
        return self.active

    def activate(self):
        if self.dateAndTime <= datetime.now().replace(second=0, microsecond=0):
            self.dateAndTime = self.dateAndTime + timedelta(days=1)

        self.setActive(True)

    def deactivate(self):
        self.setActive(False)



