from pynput.keyboard import Key, Controller
from subprocess import call

PRINT_ONLY = False

class PresentationController:
    def __init__(self):
        self.NUM_OF_REPEATED_GESTURES_TO_MAKE_A_MOVE = 3
        self.VOLUME_STEP = 10
        self.keyboard = Controller()
        self.gesture_queue = []
        self.volume = 50
        pass

    def add_gesture(self, gesture: str):
        self.gesture_queue.append(gesture)
        if len(self.gesture_queue) >= self.NUM_OF_REPEATED_GESTURES_TO_MAKE_A_MOVE:
            self.process_gesture_queue()

    def process_gesture_queue(self):
        if self.check_if_all_elements_in_list_equal(self.gesture_queue[-self.NUM_OF_REPEATED_GESTURES_TO_MAKE_A_MOVE:]):
            self.apply_move(self.gesture_queue[len(self.gesture_queue) - 1])
            self.gesture_queue = []

    def check_if_all_elements_in_list_equal(self, list_to_check):
        return len(set(list_to_check)) <= 1

    def apply_move(self, move_to_apply: str):
        print("\nmove_to_apply =", move_to_apply)
        if PRINT_ONLY:
            return

        if move_to_apply == 'right':
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)
        elif move_to_apply == 'left':
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.left)
        elif move_to_apply == 'play':
            self.keyboard.press('k')
            self.keyboard.release('k')
        elif move_to_apply == 'volume_up':
            print('Volume up !!!')
            self.raise_volume()
            # self.keyboard.press(Key.media_volume_up)
            pass
        elif move_to_apply == 'volume_down':
            self.lower_volume()
            # self.keyboard.press(Key.media_volume_down)
        pass

    def lower_volume(self):
        if self.volume > 4:
            self.volume = self.volume - self.VOLUME_STEP
            self.set_volume(self.volume)

    def raise_volume(self):
        if self.volume < 96:
            self.volume = self.volume + self.VOLUME_STEP
            self.set_volume(self.volume)

    def set_volume(self, volume):
        try:
            if (volume <= 100) and (volume >= 0):
                call(["amixer", "-D", "pulse", "sset", "Master", str(volume) + "%"])
                print("Set volume")
                valid = True

        except ValueError as error:
            print(error)
