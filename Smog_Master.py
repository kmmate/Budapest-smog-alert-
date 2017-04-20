
######### MASTER CODE to Project Smog ###############

#Settings to be modified in PyCharm:
# 1) settings/build.../console/python console
# 2) source root
# 3) add the paths to the interpreter
# 4) add '__init__.py' blank file to subfolders

# SciPy installation:
# 1) go to http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
# 2) place the donwloaded file where the python is ("C:\Users\Máté\AppData\Local\Programs\Python\Python36")
# 3) open windows command line as administrator and run:
#  "C:\Users\Máté\AppData\Local\Programs\Python\Python36\Scripts\pip.exe" install
#  "C:\Users\Máté\AppData\Local\Programs\Python\Python36\scipy-0.19.0-cp36-cp36m-win_amd64.whl"
# for NumPy+MKL do the same steps with appropriate filenames

# For LaTeX compatibility install doublestroke package as described in 'imputation.py'



# Measuring time
import time
start_time_master=time.time()


# Running dataimport.py which imports the PM10 data from Excel file, sorts it by stations and dates
# and dropping stations. Export the DataFrame to excel file. Imports meteorological data, does preliminary cleaning,
# export to excel file
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\dataimport.py", 'rb').read(), globals())

# Running geodistance.py which produces the distance matrix (and extended distance matrix with meteorological data
# station included) and creates inverse-distance weighting functions
# Export the matrix into excel file
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\geodistance.py", 'rb').read(), globals())

# Running periods_and_descriptives.py which dissections the PM10 data into Training, Test and Evaluation periods.
# Print the descriptive statistics into latex table string. Export the periods into excel file
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\periods_and_descriptives.py", 'rb').read(), globals())

# Running descriptives_weather.py which vleans meteorological data, imputes missing values, produces descriptives
# and plots
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\descriptives_weather.py", 'rb').read(), globals())

# Running imputation.py, which imputes the missing values in PM10
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\imputation.py", 'rb').read(), globals())

# Running training.py, which produces design matrices, and trains the neural network in neuralnet.py
# iff training_mode=True (~2.5 hour) -- by default training_mode=False, so only the design matrices are created
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\training.py", 'rb').read(), globals())

# Running inference.py which makes predictions (yhat,uhat), plot for residuals
exec(open(r"C:\Users\Máté\Dropbox\CEU\2017 Winter\Econ\Smog\Codes\inference.py", 'rb').read(), globals())

# Running time
print('The running time of all scripts is', time.time()-start_time_master,'\n')