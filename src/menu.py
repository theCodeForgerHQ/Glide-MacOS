import rumps
from control import toggle
from gestures import start_gesture_thread

class GlideApp(rumps.App):
    def __init__(self):
        super().__init__("Glide")
        self.menu = ["Enable", "Disable", "Quit"]
        start_gesture_thread()

    @rumps.clicked("Enable")
    def enable(self, _):
        toggle(True)

    @rumps.clicked("Disable")
    def disable(self, _):
        toggle(False)

    @rumps.clicked("Quit")
    def quit_app(self, _):
        rumps.quit_application()

if __name__ == "__main__":
    GlideApp().run()
