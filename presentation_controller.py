import pynput
from pynput.keyboard import Key
from subprocess import call

PRINT_ONLY = False

# screen resolution
MIN_X = 2934
MIN_Y = 10
MAX_X = 4623
MAX_Y = 1068
SCREEN_WIDTH = MAX_X - MIN_X
SCREEN_HEIGHT = MAX_Y - MIN_Y


class PresentationController:
    def __init__(self, ):
        self.NUM_OF_REPEATED_GESTURES_TO_MAKE_A_MOVE = 3
        self.VOLUME_STEP = 10
        self.keyboard = pynput.keyboard.Controller()
        self.mouse = pynput.mouse.Controller()
        self.gesture_queue = []
        self.volume = 50
        self.is_pointer_on = False
        self.pointer_relative_position = (50, 50)
        pass

    def add_gesture(self, gesture: str):
        self.gesture_queue.append(gesture)
        if len(self.gesture_queue) >= self.NUM_OF_REPEATED_GESTURES_TO_MAKE_A_MOVE:
            self.process_gesture_queue()

    def process_gesture_queue(self):
        if self.check_if_all_elements_in_list_equal(self.gesture_queue[-self.NUM_OF_REPEATED_GESTURES_TO_MAKE_A_MOVE:]):
            self.apply_move(self.gesture_queue[len(self.gesture_queue) - 1])
            self.gesture_queue = []
        elif self.is_pointer_on and self.gesture_queue[-1] == 'pointer':
            self.handle_pointer()

    def check_if_all_elements_in_list_equal(self, list_to_check):
        return len(set(list_to_check)) <= 1

    def apply_move(self, move_to_apply: str):
        print("\nmove_to_apply =", move_to_apply)
        if PRINT_ONLY:
            return

        if move_to_apply != 'pointer' and self.is_pointer_on is True:
            self.keyboard.press('l')
            self.keyboard.release('l')
            self.is_pointer_on = False

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
            self.raise_volume()
        elif move_to_apply == 'volume_down':
            self.lower_volume()
        elif move_to_apply == 'pointer':
            self.handle_pointer()

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

    def set_current_relative_pointer_pos(self, pointer_pos: tuple):
        # print("Pointer pos:", pointer_pos)
        self.pointer_relative_position = pointer_pos

    def handle_pointer(self):
        if self.is_pointer_on is False:
            self.keyboard.press('l')
            self.keyboard.release('l')
            self.is_pointer_on = not self.is_pointer_on

        x = self.pointer_relative_position[0]
        y = self.pointer_relative_position[1]
        self.mouse.position = MIN_X + (x * SCREEN_WIDTH), MIN_Y + (y * SCREEN_HEIGHT)

        pass
