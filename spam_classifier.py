import os
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# some usefull global variables 
PATH_TO_HAM_DIR = "/home/milkovic/Naive_bayes_classifier_using_python/naive-bayes-spam-classifier-machine-learning/emails/ham"
PATH_TO_SPAM_DIR = "/home/milkovic/Naive_bayes_classifier_using_python/naive-bayes-spam-classifier-machine-learning/emails/spam"

SPAM_TYPE = "SPAM"
HAM_TYPE = "NON-SPAM"

#arrays X and Y are of the same sizes
X = [] # represents the input data set in our case the mails 
#indacte if it is a mail or not 
Y = [] #labels for the training set 


def readFilesFromDirectory(path, classification):
    os.chdir(path) # change the working path 
    files_name = os.listdir(path) #list the directories in the path  
    for current_file in files_name:
        message = extract_mail_body(current_file) #read the content of the file
        X.append(message)#load the different mails that is the training dataset 
        Y.append(classification)#identificattion of the corresponding data ie if the mail is a spam or not 
       
           
#function to read text in a given file 
# Also here we take care to select exclusively the body of the mail 
# beacause we dont need the headers of the various mails
def extract_mail_body(file_name_str): # 
    inBody = False # 
    lines = []
    file_descriptor = io.open(file_name_str,'r', encoding='latin1')# open the file with the encoding format latin 1 which facilitates working with special characters 
    for line in file_descriptor:
        if inBody:
            lines.append(line)
        elif line == '\n':
            inBody = True # add the the lines to the list Lines once the body of the mail is reached 
        message = '\n'.join(lines) # join the different lines to form a complete mail 
    file_descriptor.close() # close the file 
    return message # return the messsage 

# here we call the function to load the mails first we load the non spam mails then the spam mails 
readFilesFromDirectory(PATH_TO_HAM_DIR, HAM_TYPE)
readFilesFromDirectory(PATH_TO_SPAM_DIR, SPAM_TYPE)

training_set = pd.DataFrame({'X': X, 'Y': Y})


#------------------


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(training_set['X'].values)

classifier = MultinomialNB()
targets = training_set['Y'].values
classifier.fit(counts, targets)


examples = ["Free Viagra now!!!", "Hi Bob, how about a game of golf tomorrow?","Hello can i meet you tomorrow?","5) Start Your Private Photo Album Online! http://www.adclick.ws/p.cfm?o=283&s=pk007"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
