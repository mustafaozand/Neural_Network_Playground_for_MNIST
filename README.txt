Neural Network Visualisation Project

This project is a neural network visualisation tool that I made to help users understand what is happening inside a neural network while it is training, and how its performance is affected by different hyperparameter configurations.

Instead of only seeing the final output, users can see how the network changes over time, how the predictions improve, and how different parts of the model behave during training.

How to set up the environment

1. Open the terminal.

2. Go to the project folder:
cd "Name used to save the project"

3. Create a virtual environment:
python3 -m venv .venv

4. Activate the virtual environment:
source .venv/bin/activate

5. Install the libraries needed for the project, using the requirements.txt:
(Make sure that the python version is 3.13, or else you might get unexpected errors)
pip install -r requirements.txt

6. Run the main program:
python main.py

Important note

Do not run the Jupyter notebook like this:
python nn_from_scratch.ipynb

That notebook is for development and testing, so it should be opened in Jupyter or inside PyCharm.
