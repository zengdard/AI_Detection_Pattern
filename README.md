StendhalGPT
===========

StendhalGPT is a web application that allows users to input two texts and compare them using various text analysis metrics. The application is built using Streamlit and uses the Natural Language Toolkit (nltk) library for text processing and analysis.

Features
--------

* Calculates lexical, grammatical, and verbal richness of texts
* Compares texts using Markov models and KL divergence
* Visualizes text analysis results using 2D and 3D plots
* Generates text using OpenAI's GPT-3 language model

Requirements
------------

* Python 3.7+
* Streamlit
* nltk
* NumPy
* SciPy
* Matplotlib
* OpenAI API key (for text generation)

Installation
------------

1. Clone the StendhalGPT repository:
```bash
git clone https://github.com/your-username/stendhalgpt.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
```
Usage
-----

1. Run the StendhalGPT application:
```
streamlit run app.py
```
2. Enter two texts in the provided text areas.
3. Click the "Check" button to analyze and compare the texts.
4. Explore the various text analysis metrics and visualizations.

Contributing
------------

Contributions to StendhalGPT are welcome! Please open an issue or a pull request if you have any suggestions or improvements.

License
-------

StendhalGPT is licensed under the MIT License.
