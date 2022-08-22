"""
MORSE CODE TRANSLATOR
By: Ogunleke Samuel Ayomide
On: 21ST MAY, 2021 :: 11:36AM
YouTube Video Demo: https://youtu.be/yxRsvIN2EZI (Please subscribe)

I came about this idea to bring Morse Code Translators into the next level.
You can translate from Morse Code to English and Vice Versa and even listen to the sound.
########
Features:
------------
> Translate from English to Morse Code
> Translate from Morse Code to English
> Listen to the Morse Code Sound
> Listen to the English Speech

A future version will include many other languages aside English and this will be available for
almost anyone to use.

PS: I didn't create the Morse Code, Samuel Morse did. We just coincidentally have the same
name.
"""

#Imports

# App class
from kivymd.app import MDApp
# To get the KV string from a file
from kivy.lang import Builder
# Google Text-to-speech Engine
from gtts import gTTS
# To load sounds
from kivy.core.audio import SoundLoader
# To set the window size
from kivy.core.window import Window

# Set the window size to a typical Android phone size
Window.size = (340, 620)


#Classes

# Create the app class
class MCTranslatorApp(MDApp):
    # Set the currently playing sound to nothing
    playing_currently = ["Nil"]

    # Initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get the KV file
        self.screen = Builder.load_file("main.kv")

    # Build the screen
    def build(self):
        return self.screen

    def translate2Eng(self, inputText):
        # Create a dictionary with Morse codes as keys and English characters as values
        english = {
           ".-": "a",
           "-...": "b",
           "-.-.": "c",
           "-..": "d",
           ".": "e",
           "..-.": "f",
           "--.": "g",
           "....": "h",
           "..": "i",
           ".---": "j",
           "-.-": "k",
           ".-..": "l",
           "--": "m",
           "-.": "n",
           "---": "o",
           ".--.": "p",
           "--.-": "q",
           ".-.": "r",
           "...": "s",
           "-": "t",
           "..-": "u",
           "...-": "v",
           ".--": "w",
           "-..-": "x",
           "-.--": "y",
           "--..": "z",
           ".----": "1",
           "..---": "2",
           "...--": "3",
           "....-": "4",
           ".....": "5",
           "-....": "6",
           "--...": "7",
           "---..": "8",
           "----.": "9",
           "-----": "0",
           "..--..": "?",
           "--..--": ",",
           ".--.-.": "@",
           "-..-.": "/",
           }
##        print(inputText)
        if any(mc in inputText for mc in english.keys()):
            # Separate the sentence into words
            inputText = inputText.split(' ')
            # For every key (Morse Code) in the dictionary above
            for t in english.keys():
                # If the key is in the input text
                if t in inputText:
                    # Get the corresponding value
                    nt = english[t]
                    # Replace every word in the `inputText` list recursively until every single character
                    # has been replaced
                    inputText[inputText.index(t)] = nt
                    # Call the function again
                    return self.translate2Eng(' '.join(inputText))      
        else:
            # If all the words/characters have been replaced, return the output
            return inputText

    def translate(self):
        # Create a dictionary with English characters as keys and morse codes as values
        morse = {
           "a": ".-",
           "b": "-...",
           "c": "-.-.",
           "d": "-..",
           "e": ".",
           "f": "..-.",
           "g": "--.",
           "h": "....",
           "i": "..",
           "j": ".---",
           "k": "-.-",
           "l": ".-..",
           "m": "--",
           "n": "-.",
           "o": "---",
           "p": ".--.",
           "q": "--.-",
           "r": ".-.",
           "s": "...",
           "t": "-",
           "u": "..-",
           "v": "...-",
           "w": ".--",
           "x": "-..-",
           "y": "-.--",
           "z": "--..",
           "1": ".----",
           "2": "..---",
           "3": "...--",
           "4": "....-",
           "5": ".....",
           "6": "-....",
           "7": "--...",
           "8": "---..",
           "9": "----.",
           "0": "-----",
           "?": "..--..",
           ",": "--..--",
           "@": ".--.-.",
           "/": "-..-."
           }
        # Convert the input text into lowercase and storing it in a variable
        inputText = self.screen.ids.inp.text.lower()
        # Setting the default output text to an empty string
        outputText = ""
        # Getting the indicator, whether it's from Morse to English or otherwise
        indicator = self.screen.ids.ddi.current_item
        # If user is translating from English to Morse
        if indicator == "English": # to Morse
            # For every key in the dictionary above
            for t in morse.keys():
                # If the key is in the input text
                if t in inputText:
                    # Get the value of that key
                    nt = morse[t]
                    # Replace all the English characters with the corresponding code and a whitespace
                    inputText = inputText.replace(t, nt+" ")
                    # Set the `outputText` variable into the outcome from `inputText`
                    outputText = inputText
            # Display the result
            self.screen.ids.res.text = outputText

        else: # to English

            # TRICKY ALGO HERE...
            try:
                res = self.translate2Eng(inputText)
                # Replace all whitespaces with a dot
                res = res.replace(" ", ".")
                # Replace every occurrence of two dots with a single whitespace
                res = res.replace("..", " ")
                # Replace every occurence of one dot with an empty string
                res = res.replace(".", "")
                # Set the output text to the outcome from `res`
                self.screen.ids.res.text = res
            except:
                # A warning here...
                print("An error occured! Don't try again with the same input")


            try:
                # Record the speech of the translated English word/sentence
                tts = gTTS(self.screen.ids.res.text, lang="en-us")
                # Save it to an mp3 file
                tts.save("tts.mp3")
            except:
                print("Error!")


    def speak(self):
        # If user wants to translate from English
        if self.screen.ids.ddi.current_item == "English":
            import glob
            # Create an empty dictionary
            morseCodeAlpha = {}
            import string
            # Store all UPPERCASE letters in the `letters` variable
            letters = string.ascii_uppercase

            # For every audio file in the Morse-Code-Alphabets folder
            for idx, file in enumerate(glob.glob("Morse-Code-Alphabets/*.mp3")):
                # Add the keys as the letters and values as corresponding audio files
                morseCodeAlpha[letters[idx]] = file

            # Converting all the English words into UPPERCASE
            test = self.screen.ids.inp.text.upper()

            # Importing time as a local module
            import time

            # For every letter in the English sentence/word
            for t in test:
                # If the letter is in a morse code alphabet sound filename
                if t in morseCodeAlpha.keys():
                    try:
                        # Load that sound file
                        sound = SoundLoader.load(morseCodeAlpha[t])
                        if sound:
                            # Set it to the currently playing sound
                            self.playing_currently.append(t)
    ##                        print("Fine!")
                            # Play it
                            sound.play()
                            # Wait for 2 seconds after which the sound must have finished playing
                            time.sleep(2)
                            # Stop the sound
                            sound.stop()
                     # If there is an error
                    except:
                         print("Error!")
            print("Okay")
        else:
            print("Translation is from Morse to English")

            # Load the saved audio gTTS translation
            sound = SoundLoader.load("tts.mp3")

            # If loaded correctly
            if sound:
                # Play the sound
                sound.play()

    def update_current(self):
        # Set the currently playing sound
        self.screen.ids.playing.text = self.playing_currently[-1]



#Driver code
if __name__ == "__main__":
    # Run an instance of the app class
    MCTranslatorApp().run()
