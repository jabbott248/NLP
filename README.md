# NLP

Hello!

This is a collection of Python scripts exploring natural language processing, including:

- **[Spell checker](SpellChecker/)**: a spell checker that uses a trie structure to efficiently check if a word is spelt correctly.
- **[Author language model](Author_Language_Model/)**: a script that creates a language model based off passed-in text, that can be used to determine the author of new text.
- **[Part-of-speech tagger](Part_Of_Speech_Tagger/)**: a part-of-speech tagger that can determine which part of speech a word is.
- **[Audio spectrogram](Audio_Spectrogram/)**: a script that takes in a .wav audio file and creates a spectrogram of the audio using a Fourier transform.

## Setup

The scripts are written in Python 3 and were developed and tested with Python 3.14 on macOS.

### 1. Get the code

```bash
git clone https://github.com/jabbott248/NLP.git
cd NLP
```

### 2. Create a virtual environment and install the dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip3 install -r requirements.txt
```

This installs NLTK, NumPy, Matplotlib and Pillow. What each project needs:

| Project | Folder | Needs |
|---|---|---|
| Spell checker | `SpellChecker/` | Python standard library only |
| Author language model | `Author_Language_Model/` | NLTK + its `punkt` tokenizer data |
| Part-of-speech tagger | `Part_Of_Speech_Tagger/` | NLTK + its `brown`, `words` and `punkt` data |
| Audio spectrogram | `Audio_Spectrogram/` | NumPy and Matplotlib |

### 3. macOS only: install Python's certificates

If Python came from the python.org installer, it can't verify SSL certificates until you run its bundled certificate installer, and NLTK's downloads fail with `CERTIFICATE_VERIFY_FAILED`. Run this once, changing the version to match yours:

```bash
open "/Applications/Python 3.14/Install Certificates.command"
```

Python from Homebrew or a Linux package manager doesn't need this.

### 4. Download the NLTK data

The NLTK scripts download any data they're missing (to `~/nltk_data`) the first time they run, so this step is optional. To download it all up front:

```bash
python3 -c "import nltk; [nltk.download(p) for p in ('brown', 'words', 'punkt', 'punkt_tab')]"
```

### 5. Check that everything works

Each project with tests can be checked from its own folder:

```bash
(cd SpellChecker && python3 -m unittest)            # 49 tests, about 5 s
(cd Author_Language_Model && python3 -m unittest)   # 62 tests, about 13 s
(cd Audio_Spectrogram && python3 -m unittest)       # 49 tests, about 5 s
```

Each folder has its own README (`*_README.txt`) explaining how the project works and how to run it.

Best,

Jake Abbott
