# AdOpinion

A minimalistic advertisement algorithm powered by Sentiment Analysis 

![AdOpinion : retrieve the list of users to target by brand](doc/Intro_AdOpinion.gif)
## Installation 

* First, install [node](https://nodejs.org/en/)
* Then, enter in the terminal : `git clone https://github.com/gabrielmougard/AdOpinion.git && cd AdOpinion && npm install`
* Run `pip3 install -r requirement.txt` to install python3 dependencies.
* Install some dictionnaries : type `python3` in a terminal then enter in the interpreter :
`import nltk` and `nltk.download()` then type `d` and enter the following keywords : `stopwords`, `punkt`, `averaged_perceptron_tagger`, `wordnet`.
* Finally, execute `npm start` in the same terminal and open your web browser where you can type `localhost:3000` to open the platform !

## Important note
* Wait a little bit when you want to click on the two buttons because you have to wait for the predictions (10s max)! 

Cheers !
