# Dota2Predictor
  
This is the final project of CSC576, Advanced Machine Learning class, where team of three did this project together.  
The basic intention of this project is to build a predictive model about Dota2 with:  
Input: detailed game stats of each player (including K/A/D, Gold/min, Exp/min, hero used, etc.)  
Output: A model that contains two parts 1. model of each users archived 2. model of the compatibility of each team (5 heros)
  
Goal: based on the two outputs above, the team wishes to predict the result of each game at the "very begining" (before users choose their heros) and "after hero choose but before clock begin".
 
Code:  
Team constructed a tensorflow model with a cost function and achieved 97% accuracy on training data. Need further improvment (shall start recently 3/20/2017)
