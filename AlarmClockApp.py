from kivy.app import App
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.core.window import Window
from kivy.core.audio import SoundLoader
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
from kivy.uix.progressbar import ProgressBar
from kivy.properties import ListProperty, StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from AlarmManager import AlarmManager
from FaceRecognitionAPI import FaceRecognitionAPI
from OpticalFlowController import OpticalFlowController
from multiprocessing import Process, Queue
from datetime import datetime, timedelta
import cv2
import os

video_ind = 1
frame_width = 350
frame_height = 350
screenshots_dir = './screenshots'
sound_dir = './alarms/funny.mp3'
face_dir = './faces'
weight_dir = './MobileFaceNet_Pytorch/model/best/068.ckpt'
haar_dir = './Siamese_MobileNetV2/src/utils/haarcascade_frontalface_default.xml'
alarms_dir = './alarms/alarms'
sound = SoundLoader.load(sound_dir)
sound.loop = True
manager = AlarmManager(alarms_dir)
sm = ScreenManager()
facerecognition = FaceRecognitionAPI(face_dir, weight_dir, haar_dir)
opticalflowcontroller = OpticalFlowController(video_ind, frame_width=frame_width, frame_height=frame_height)

class HomeWindow(Screen): #Alarm list, button to re-register face, button to trigger add new alarm (disabled if face unregistered)
                          #Popup with slider to set time, button to set alarm, cross button to exit
                          #Button to trigger night mode (hide everything except alarm, dim screen)
                          #Night mode turned off on touch
    global sm
    global manager
    global facerecognition
    screenmanager = sm
    alarmmanager = manager
    facerecog = facerecognition

    def checkAndTrigger(self, dt):
        triggered, triggered_alarms = self.alarmmanager.checkAndTriggerAlarms()
        if triggered:
            alarm = triggered_alarms[0]
            index = alarm.getIndex()
            time, notation = alarm.get12()
            label = alarm.getLabel()
            self.screenmanager.get_screen('AlarmWindow').setValues(index, time, notation, label)
            self.screenmanager.get_screen('AlarmWindow').trigger()
            self.screenmanager.transition.direction = 'down'
            self.screenmanager.current = 'AlarmWindow'

class RegisterWindow(Screen): #Button to start face registration on press
                              #Instructions to guide registration
    global sm
    global facerecognition
    global opticalflowcontroller
    screenmanager = sm
    facerecog = facerecognition
    opticalflow = opticalflowcontroller
    start = ObjectProperty()
    stream = ObjectProperty()
    countdown = ObjectProperty()
    slowdown = ObjectProperty()
    progress_bar = ObjectProperty()

    def __init__(self, **kwargs):
        super(RegisterWindow, self).__init__(**kwargs)
        self.prev = None
        self.screencaps = []
        self.registering = False
        self.num_images_required = 20
        self.valid_images = 0
        self.progress_bar.value_normalized = 0.0
        self.stream_event = None
        self.capture_event = None

    def startStream(self):
        self.valid_images = 0
        self.progress_bar.value_normalized = 0.0
        self.opticalflow.start()
        self.stream_event = Clock.schedule_interval(self.update, 1.0/33.0)

    def endStream(self):
        if self.stream_event is not None:
            self.stream_event.cancel()

    def endCapture(self):
        if self.capture_event is not None:
            self.capture_event.cancel()

    def startRegistration(self):
        self.registering = True
        self.capture_event = Clock.schedule_interval(self.capture, 1.0/2.0)

    def completeRegistration(self):
        self.registering = False
        self.endCapture()
        Clock.schedule_once(self.endRegistration, 2.5)
        self.register()

    def endRegistration(self, dt):
        self.endStream()
        self.goHome()
        self.opticalflow.release()
        self.start.disabled = False

    def cancelRegistration(self):
        self.registering = False
        self.endCapture()
        self.endRegistration(None)

    def register(self):
        face_id = self.facerecog.batchRegister(self.screencaps)
        self.facerecog.saveCurrent(face_id) #Set face as current

    def goHome(self):
        self.screenmanager.transition.direction = 'down'
        self.screenmanager.current = 'HomeWindow'

    def update(self, dt):
        self.frame, self.prev, overLimit = self.opticalflow.step(self.prev)
        if overLimit and self.registering:
            self.slowdown.start(0.75)

        buf = cv2.flip(self.frame, 0)
        buf = buf.tostring()
        streamTexture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        streamTexture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.stream.texture = streamTexture

    def capture(self, dt): #capture every 0.5 s
        valid, _, input_image, _ = self.facerecog.cropAndPreprocessFrame(self.frame)
        if valid:
            self.valid_images += 1
            self.screencaps.append(input_image)
            self.updateProgressBar()

    def updateProgressBar(self):
        self.progress_bar.value_normalized = self.valid_images / self.num_images_required
        if self.progress_bar.value_normalized == 1.0:
            self.completeRegistration()

class RegisterProgressBar(ProgressBar):
    pass


class AlarmWindow(Screen): #Upon recognition a pop-up to ask if snooze or turn off, track stats in background
    global facerecognition
    global sm
    global manager
    alarmmanager = manager
    screenmanager = sm
    facerecog = facerecognition
    stream = ObjectProperty()
    alarmlabel = ObjectProperty()

    def trigger(self):
        global video_ind
        global frame_width
        global frame_height
        global sound
        self.cap = cv2.VideoCapture(video_ind)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.detected = False
        self.p = None
        self.queue = Queue()
        self.stream_event = Clock.schedule_interval(self.update, 1.0/33.0)
        self.alarmlabel.start(1)
        sound.play()

    def stop(self):
        self.cap.release()
        self.alarmlabel.finish()
        self.stream_event.cancel()
        sound.stop()
        if self.p is not None:
            self.p.terminate()

    def setValues(self, idx, time, notation, label):
        self.curr_idx = idx
        self.time = time
        self.label = label
        self.alarmmanager.deactivateAlarm(idx)

    def update(self, dt):
        _, frame = self.cap.read()
        frame = self.flipFrame(frame)
        if not self.detected:
            frame = self.detectFace(frame)
        buf = cv2.flip(frame, 0)
        buf = buf.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.stream.texture = texture

    def detectFace(self, frame):
        detected, processed_frame, image, self.screenshot = self.facerecog.cropAndPreprocessFrame(frame)
        if detected:
            if self.p is not None:
                if self.p.is_alive():
                    pass
                else:
                    retval = self.queue.get()
                    if retval:
                        self.closeAlarm()
                    self.p = None
            else:
                self.p = Process(target=self.identifyFace, args=(self.queue, image,))
                self.p.start()

        return processed_frame

    def closeAlarm(self):
        self.stop()
        self.alarmmanager.deactivateAlarm(self.curr_idx)
        self.screenmanager.get_screen('PostAlarmWindow').initialize(self.screenshot)
        self.screenmanager.transition.direction = 'left'
        self.screenmanager.current = 'PostAlarmWindow'

    def identifyFace(self, queue, image):
        res = self.facerecog.runRecognition(image)
        queue.put(res)

    def flipFrame(self, frame):
        return cv2.flip(frame, 1)

class PostAlarmWindow(Screen): #Upon turn off, greet good morning, good afternoon etc. according to time
                                #Show stats: Time taken to turn off alarm, no. of times snoozed, high score, running average and stuff
    global sm
    screenmanager = sm
    screencap = ObjectProperty()

    def initialize(self, image):
        self.image = image
        buf = cv2.flip(image, 0)
        buf = buf.tostring()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.screencap.texture = texture

    def goHome(self):
        self.screenmanager.transition.direction = 'up'
        self.screenmanager.current = 'HomeWindow'

    def saveImage(self):
        global screenshots_dir
        timestamp = datetime.now().strftime("%y_%m_%d-%I_%M_%S_%p.jpg")
        cv2.imwrite(os.path.join(screenshots_dir, timestamp), self.image)

class EditAlarmWindow(Screen):
    index = NumericProperty()
    hour = ObjectProperty()
    minute = ObjectProperty()
    notation = ObjectProperty()
    label = ObjectProperty()

    def setValues(self, idx, hour, minute, notation, label):
        self.index = idx
        self.hour.text = hour
        self.minute.text = minute
        self.notation.text = notation
        self.label.text = label

    def getValues(self):
        return self.index, self.hour.text, self.minute.text, self.notation.text, self.label.text

class WindowsManager(ScreenManager):
    pass

class AlarmList(RecycleView): #Controller
    alarmlayout = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(AlarmList, self).__init__(**kwargs)
        global manager
        self.manager = manager
        self.reload_event = Clock.schedule_interval(self.reload, 0.2)
        self.edit = False

    def reload(self, dt):
        self.data = [self.getAlarmDict(idx) for idx in range(len(self.manager))]

    def getAlarmDict(self, idx):
        alarm = self.manager.getAlarm(idx)
        time, notation = alarm.get12()
        label = alarm.getLabel()
        active = alarm.isActive()
        return {'index': idx, 'time': time, 'notation': notation, 'label': label, 'active': active}

    def editAlarm(self, index, hour, minute, notation, label):
        self.reload_event.cancel()
        manager.editAlarm(index, hour, minute, notation, label)
        manager.saveAlarms(alarms_dir)
        self.reload_event = Clock.schedule_interval(self.reload, 0.2)

    def addAlarm(self, hour, minute, notation, label):
        self.reload_event.cancel()
        index = manager.addAlarm(hour, minute, notation, label)
        manager.saveAlarms(alarms_dir)
        self.reload_event = Clock.schedule_interval(self.reload, 0.2)

    def removeAlarm(self, index):
        self.reload_event.cancel()
        manager.removeAlarm(index)
        manager.saveAlarms(alarms_dir)
        self.reload_event = Clock.schedule_interval(self.reload, 0.2)

    def activateAlarm(self, idx):
        self.reload_event.cancel()
        manager.activateAlarm(idx)
        manager.saveAlarms(alarms_dir)
        self.reload_event = Clock.schedule_interval(self.reload, 0.2)

    def deactivateAlarm(self, idx):
        self.reload_event.cancel()
        manager.deactivateAlarm(idx)
        manager.saveAlarms(alarms_dir)
        self.reload_event = Clock.schedule_interval(self.reload, 0.2)

    def toggleEdit(self):
        self.edit = not self.edit

    def isEditActive(self):
        return self.edit

class AlarmLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):  #View
    pass

alarm_list = AlarmList()

class SelectableAlarm(RecycleDataViewBehavior, BoxLayout):
    global sm
    global alarm_list
    screenmanager = sm
    alarmlist = alarm_list
    index = NumericProperty()
    time = StringProperty()
    notation = StringProperty()
    label = StringProperty()
    active = BooleanProperty()

    def __init__(self, **kwargs):
        super(SelectableAlarm, self).__init__(**kwargs)

    def switchAlarm(self, active):
        if not self.alarmlist.isEditActive():
            if active:
                self.alarmlist.activateAlarm(self.index)
            else:
                self.alarmlist.deactivateAlarm(self.index)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if self.alarmlist.isEditActive():
                alarm = manager.getAlarm(self.index)
                hour = alarm.getHour()
                minute = alarm.getMinute()
                notation = alarm.getNotation()
                label = alarm.getLabel()
                self.screenmanager.get_screen('EditAlarmWindow').setValues(self.index, hour, minute, notation, label)
                self.screenmanager.transition.direction = 'up'
                self.screenmanager.current = 'EditAlarmWindow'

                return True

        return False

class AlarmWindowLabel(Label):

    def start(self, duration):
        Animation.cancel_all(self)  # stop any current animations
        self.anim = Animation(opacity=1, duration=duration*0.5) + Animation(opacity=0, duration=duration*0.5)
        self.anim.repeat = True
        self.anim.start(self)

    def finish(self):
        self.anim.stop(self)

class AlarmListLabel(Label):
    pass

class SlowDownLabel(Label):

    def start(self, duration):
        Animation.cancel_all(self)  # stop any current animations
        self.anim = Animation(opacity=1, duration=duration*0.25) + Animation(opacity=0, duration=duration*0.75)
        self.anim.start(self)

class EditAlarmsButton(Label):
    global alarm_list
    alarmlist = alarm_list

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 0.5

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): 
            self.opacity = 1.0

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 1
            self.alarmlist.toggleEdit()
            if self.alarmlist.isEditActive():
                self.text = 'Done'
            else:
                self.text = 'Edit'

class RegisterFaceButton(Button):
    global sm
    global alarm_list
    screenmanager = sm
    alarmlist = alarm_list
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
            self.screenmanager.get_screen('RegisterWindow').startStream()
            self.screenmanager.transition.direction = 'up'
            self.screenmanager.current = 'RegisterWindow'

class AddAlarmButton(Button):
    global sm
    global alarm_list
    screenmanager = sm
    alarmlist = alarm_list
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
            nearest_five = self.roundUp(datetime.now())
            hour = nearest_five.strftime("%I")
            minute = nearest_five.strftime("%M")
            notation = nearest_five.strftime("%p")
            self.screenmanager.get_screen('EditAlarmWindow').setValues(-1, hour, minute, notation, 'Alarm') #Reset values
            self.screenmanager.transition.direction = 'up'
            self.screenmanager.current = 'EditAlarmWindow'

    def roundUp(self, dt, minutes=5):
        return dt + (datetime.min - dt) % timedelta(minutes=minutes)

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
    global alarm_list
    screenmanager = sm
    opacity = NumericProperty(1)
    alarmlist = alarm_list

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
            self.handleChanges(index, hour, minute, notation, label)
            self.screenmanager.transition.direction = 'down'
            self.screenmanager.current = 'HomeWindow'

    def handleChanges(self, index, hour, minute, notation, label):
        if not hour or not minute or not notation:
            now = datetime.now()
            hour = now.strftime('%I')
            minute = now.strftime('%M')
            notation = now.strftime('%p')

        if not label:
            label = 'Alarm'

        if index < 0: #Perform changes in manager, then sync data
            self.alarmlist.addAlarm(hour, minute, notation, label)
        else:
            self.alarmlist.editAlarm(index, hour, minute, notation, label)

class DeleteEditAlarm(Label):
    global manager
    global sm
    global alarm_list
    screenmanager = sm
    opacity = NumericProperty(1)
    alarmlist = alarm_list

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 0.5

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos): 
            self.opacity = 1.0

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos): 
            self.opacity = 1
            index, _, _, _, _ = self.screenmanager.get_screen('EditAlarmWindow').getValues()
            if index >= 0:
                self.alarmlist.removeAlarm(index)
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
        print('SWITCH')
        if self.collide_point(*touch.pos):
            self.parent.switchAlarm(not self.active)

            return True

        return False


class TimeBar(Label):
    background_color = ListProperty((0,0,0,1))
    def __init__(self, **kwargs):
        super(TimeBar, self).__init__(**kwargs)
        self.text = datetime.now().strftime("%I:%M %p")

    def updateTime(self, dt):
        self.text = datetime.now().strftime("%I:%M %p")

class AlarmClockLayout(BoxLayout):
    pass

class AlarmClockApp(App):

    def build(self):
        global sm
        global alarm_list
        homewindow = HomeWindow(name='HomeWindow')
        homewindow.add_widget(alarm_list)
        sm.add_widget(homewindow)
        sm.add_widget(EditAlarmWindow(name='EditAlarmWindow'))
        sm.add_widget(RegisterWindow(name='RegisterWindow'))
        sm.add_widget(AlarmWindow(name='AlarmWindow'))
        sm.add_widget(PostAlarmWindow(name='PostAlarmWindow'))
        time_bar = TimeBar(size_hint=(1, 0.05))
        Clock.schedule_interval(time_bar.updateTime, 1)
        Clock.schedule_interval(homewindow.checkAndTrigger, 1)
        mainlayout = AlarmClockLayout(orientation='vertical')
        mainlayout.add_widget(time_bar)
        mainlayout.add_widget(sm)
        Window.bind(on_request_close=self.on_request_close)
        return mainlayout

    def on_request_close(self, *args, **kwargs):
        global opticalflowcontroller
        global manager
        global alarms_dir
        global sound
        manager.saveAlarms(alarms_dir)
        opticalflowcontroller.release()
        sound.unload()
        return False

if __name__ == '__main__':
    #Window.fullscreen = True
    AlarmClockApp().run()
    cv2.destroyAllWindows()