Higgs Boson's Challenge - Group 70
----------------------------------

How to run:

- The data should be in a folder /data
- The output is written in a folder /output

Submission content :

The current folder contains three folders:

- scripts/
	List of used files : 
		Python files :
		
		- costs.py --- This file contains the costs functions compute_loss, calculate_nll
		- cross_validation.py --- This file contains all the necesary to run 
			a cross validation on some model
		- helpers.py --- This helper file contains some usefull methods like the 
			standardization method or the polynomial builder for some tests.
		- proj1_helpers.py --- Given file used to load the data and create the submission essentially
		- least_squares.py --- Contains all least squares's relative functions that we use in our code
		- regression.py --- Contains all the regression's relative functions that we use in our code
		
	List of files for submission:
		- run.py --- Is the file that we can run to have our last submission output
		- implementation.py --- The file contains the 6 methods we had to implement 
			at the end of the file. But it also contains all the necesary functions (like sigmoid, or polynomialBasis) for our functions
			to run correctly.
	
- data/
	This folder, as indicated by its name contains the data for the higgs boson challenge :
	- test.csv	--- The testing set for which we should predict the outputs
	- train.csv	--- The training set on which we train our model

- output/
	This folder contains the output of our model prediction :
	- out.csv --- File containing our best prediction output


NB: This project was done by Audrey Loeffel, Joachim Huet and for the last 2 hours 
before the submission deadline, the third person of the group - Yalan Yiue - finally came add 2 or 3 
sentences in the report... 


