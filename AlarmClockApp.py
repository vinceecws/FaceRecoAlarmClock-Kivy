from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.datamodel import RecycleDataModel
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.button import Button
from kivy.uix.switch import Switch
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.actionbar import ActionBar
from kivy.properties import ListProperty, StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from AlarmManager import AlarmManager
from datetime import datetime
import cv2

manager = AlarmManager('./alarms/alarms')
sm = ScreenManager()

class HomeWindow(Screen): #Alarm list, button to re-register face, button to trigger add new alarm (disabled if face unregistered)
                          #Popup with slider to set time, button to set alarm, cross button to exit
                          #Button to trigger night mode (hide everything except alarm, dim screen)
                          #Night mode turned off on touch
    pass

class RegisterWindow(Screen): #Button to start face registration on press
                              #Instructions to guide registration
    pass

class AlarmWindow(Screen): #Upon recognition a pop-up to ask if snooze or turn off, track stats in background
    pass

class PostAlarmWindow(Screen): #Upon turn off, greet good morning, good afternoon etc. according to time
                                #Show stats: Time taken to turn off alarm, no. of times snoozed, high score, running average and stuff

    pass

class EditAlarmWindow(Screen):
    curr_idx = NumericProperty()
    hour = StringProperty()
    minute = StringProperty()
    notation = StringProperty()
    label = StringProperty()

    def setValues(self, idx, hour, minute, notation, label):
        self.curr_idx = idx
        self.hour = hour
        self.minute = minute
        self.notation = notation
        self.label = label

    def getValues(self):
        return self.curr_idx, self.hour, self.minute, self.notation, self.label

class WindowsManager(ScreenManager):
    pass

class AlarmListLabel(Label):
    pass

class AddAlarmButton(Button):
    global sm
    screenmanager = sm
    opacity = NumericProperty(1)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 0.5

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): 
            self.opacity = 1.0

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 1
            self.screenmanager.transition.direction = 'down'
            self.screenmanager.current = 'HomeWindow'

class CancelEditAlarm(Label):
    global sm
    screenmanager = sm
    opacity = NumericProperty(1)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 0.5

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): 
            self.opacity = 1.0

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 1
            self.screenmanager.get_screen('EditAlarmWindow').setValues(-1, '','','','') #Reset values
            self.screenmanager.transition.direction = 'down'
            self.screenmanager.current = 'HomeWindow'

class SaveEditAlarm(Label):
    global manager
    global sm
    screenmanager = sm
    opacity = NumericProperty(1)
    alarm_list = ObjectProperty()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 0.5

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): 
            self.opacity = 1.0

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 1
            index, hour, minute, notation, label = self.screenmanager.get_screen('EditAlarmWindow').getValues()
            self.alarm_list.editAlarm(index, hour, minute, notation, label)
            self.screenmanager.transition.direction = 'down'
            self.screenmanager.current = 'HomeWindow'

class LabelTextInput(TextInput):
    pass

class LabelClear(Label):

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 0.5

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): 
            self.opacity = 1.0

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 1
            labeltextinput = self.parent.children[2]
            labeltextinput.text = ''

class AlarmSwitch(Switch):
    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            new_active = not self.active
            self.parent.switchAlarm(new_active)
            self.active = new_active
            return True

        return super(AlarmSwitch, self).on_touch_up(touch)

class AlarmList(RecycleView, RecycleDataModel):
    alarmlist_layout = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(AlarmList, self).__init__(**kwargs)
        self.manager = manager
        self.data = [self.getAlarmDict(i) for i in range(len(self.manager))]

    def getAlarmDict(self, idx):
        alarm = self.manager.getAlarm(idx)
        time, notation = alarm.get12()
        label = alarm.label
        active = alarm.isActive()

        return {'time': time, 'notation': notation, 'label': label, 'active': active}

    def editAlarm(self, index, hour, minute, notation, label):
        manager.editAlarm(index, hour, minute, notation, label)
        self.data[index] = {'time': f'{hour:02d}:{minute:02d}', 'notation': notation, 'label': label}


class SelectableAlarm(RecycleDataViewBehavior, BoxLayout):
    """ Add selection support to the Label """
    global sm
    screenmanager = sm
    index = None
    time = StringProperty()
    notation = StringProperty()
    label = StringProperty()
    active = BooleanProperty()

    def __init__(self, **kwargs):
        super(SelectableAlarm, self).__init__(**kwargs)

    def refresh_view_attrs(self, rv, index, data):
        """ Catch and handle the view changes """
        global manager
        self.index = index
        alarm = manager.getAlarm(self.index)
        self.time, self.notation = alarm.get12()
        self.label = alarm.getLabel()
        self.active = alarm.isActive()
        return super(SelectableAlarm, self).refresh_view_attrs(rv, index, data)

    def switchAlarm(self, active):
        if active:
            manager.activateAlarm(self.index)
        else:
            manager.deactivateAlarm(self.index)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            alarm = manager.getAlarm(self.index)
            hour = alarm.getHour()
            minute = alarm.getMinute()
            notation = alarm.getNotation()
            label = alarm.getLabel()
            self.screenmanager.get_screen('EditAlarmWindow').setValues(self.index, hour, minute, notation, label)
            self.screenmanager.transition.direction = 'up'
            self.screenmanager.current = 'EditAlarmWindow'

            return True

class SelectableAlarmLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    """ Adds selection and focus behaviour to the view. """
    selected_value = StringProperty('')


class TimeBar(Label):
    background_color = ListProperty((0,0,0,1))
    time = StringProperty()

    def __init__(self, **kwargs):
        super(TimeBar, self).__init__(**kwargs)
        self.time = datetime.now().strftime("%I:%M %p")

    def updateTime(self):
        self.time = datetime.now().strftime("%I:%M %p")

class AlarmClockApp(App):

    def build(self):
        global sm
        sm.add_widget(HomeWindow(name='HomeWindow'))
        sm.add_widget(EditAlarmWindow(name='EditAlarmWindow'))
        sm.add_widget(RegisterWindow(name='RegisterWindow'))
        sm.add_widget(AlarmWindow(name='AlarmWindow'))
        sm.add_widget(PostAlarmWindow(name='PostAlarmWindow'))
        return sm

if __name__ == '__main__':
    #Window.fullscreen = True
    AlarmClockApp().run()
    cv2.destroyAllWindows()
    manager.saveAlarms('./alarms/alarms')