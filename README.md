README

Hello!

This is a collection of python scripts exploring natural language processing including:

    A spell checker that uses a trie structure to effeciently check if a word is spelt correclty.
    
    A script that creates a language model based off passed in text, that can be used to determine the author of new text.
    
    A part of speech tagger that can determine which part of speech a word is.
    
    A script that takes in .wav audio file, and creates a spectrogram of the audio using a fourier transfrom

Setup

Create a virtual environment and install the dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt

Then download the NLTK data used by the scripts:

    python3 -c "import nltk; [nltk.download(p) for p in ('brown','words','punkt','punkt_tab')]"

The spectrogram viewer also needs tkinter, which pip can't install. It comes with the python.org installer; with Homebrew Python, run "brew install python-tk".

Best,

Jake Abbott
