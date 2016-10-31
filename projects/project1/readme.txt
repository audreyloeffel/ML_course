-----------
| Content :
-----------
in scripts/
-------------
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
		
		Notebooks :
		
		- project1.ipynb --- Contains a lot of unorganized code, we used it for some test and as a source of code
			for the much more organised and used files.
		- Logistic Regression --- Contains the improvements for logistic regression, the cross validation of this model
		- Ridge Regression --- Contains the analysis and improvements of the Ridge Regression model.
		- Test --- Contains a try to improve least squares but finally we changed our focus to ridge regression so not complete.
		- log_train --- Contains the pre-process features analysis. Correlations between the differents features of the data.
		- final_method  --- Contains the code that we used for the final submission, our 'optimal prediction'

	List of files for submission:
		- run.py --- Is the file that we can run to have our last submission output
		- submissions.py --- The submission file containing the 6 methods we had to implement 
			at the end of the file. But it also contains all the necesary functions for our functions
			to run correctly.
	

in data/
--------
	This folder, as indicated by it's name contains the data for the higgs boson challenge :
	- test.csv	--- The testing set for which we should predict the outputs
	- train.csv	--- The training set on which we train our model

in output/
----------
	This folder contains the output of our model prediction :
	- out.csv --- File containing our best prediction output


This project content was done by Audrey Loeffel, Joachim Huet and in the 2 last hours 
before submission deadline, the third person of the group - Yalan Yiue - finally came add 2 or 3 
words in the report. 


