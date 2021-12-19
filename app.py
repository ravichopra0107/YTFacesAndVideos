import module
import utils

if __name__ == "__main__":
    utils.wishMe()
    while True:
        query = utils.takeCommand().lower()
        if query == "open camera":
            utils.speak("Opening camera")
            module.run()