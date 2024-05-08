# DLP_PROJECT
# Error Detection In Quranic Recitation

## Project made by:
Sohaib Ashraf (20k-0488)  
Khawaja Abdullah (20k-0385)  
Muhammad Shahzaib (20k-1067)  

The Quran is being recited globally and incorrect recitation is common due to the complexity 
of the Arabic language and pronunciation. Currently, there is no widely-available automated 
system that can detect and correct errors in Quranic recitation. 
By referring to the problem statement, in this project, we developed of a high-accuracy 
system that detects errors in Quranic recitation, using the concepts of machine learning at 
the core, that provides the learner immediate feedback as to whether their recitation is 
correct. The project includes the usage of following concepts: 
- Speech Recognition. 
- Classification; recitation is correct or incorrect. 
- Error Detection. 
It will be very helpful in learning how to correctly recite Quran and allows learners to 
practice this at any time and with unlimited repetitions, without the presence of an 
instructor. 

## Methodology:

### Dataset:
The Quranic recitation dataset was obtained from two websites: 
- [Everyayah](https://www.everyayah.com/data)  
- [DailyAyat](https://dailyayat.com/)  
The recordings were collected from different reciters and contained different surahs 
and ayahs from the Quran. The Everyayah website contains a collection of recitations 
from 60 different reciters, while the DailyAyat website contains recitations from 20 
different reciters. But this data had to be cleaned before using for our model which 
reduced it to being from only 66 reciters combined. The audio files were downloaded 
in MP3 format.  

### Data Preprocessing:
To preprocess the data, the audio files were first converted to WAV format and 
resampled to a sampling rate of 22kHz. 
To increase the size of the dataset and prevent overfitting of the models, data 
augmentation techniques were applied to the original recordings. Specifically, pitch 
shifting was used to create new recordings. Pitch shifting was used to increase or 
decrease the pitch of the original recording without changing its duration. This was 
achieved by using the librosa.effects.pitch_shift function from the librosa Python 
library, which shifts the pitch of the audio signal by a specified number of semitones. 
Pitch shifting was performed by randomly selecting a semitone value between -3 and 
3 and applying it to the original recording. 
Test size was 10%, validation set was 15% and the training set was 75%.  

### Feature Extraction:
The feature extraction process involved loading an audio file using the Librosa library. 
Then, the MFCCs are computed using the librosa.feature.mfcc function, which takes 
the audio signal and sampling rate as input parameters.  
After the MFCCs are computed, they are reshaped to have the expected shape by 
creating a new array of zeros with the same number of rows as the original MFCCs 
and the same number of columns as the expected number of MFCC vectors per 
segment. Then, the original MFCCs are copied into this new array, starting from the 
first column. 
Finally, the extracted MFCCs are stored in a JSON file, along with the corresponding 
label for the audio file. This file is used as input to the machine learning model.  

### Speech-to-Text Conversion:
For this task, Google Speech-to-Text API was used. The feature extraction and 
speech recognition are done by the API itself. The API uses a pre-trained neural 
network to perform speech recognition on the audio input and returns the 
corresponding text output in Arabic. The audio file is first read using the pydub 
library, and then exported as a WAV file. The Speech-to-Text API is then called with 
the appropriate configuration, which includes enabling automatic punctuation and 
setting the language code to Arabic. The response from the API is then processed to 
extract the transcript of the audio file. Finally, the extracted transcript is appended to 
a list called 'extracted_features'. After this, some data preprocessing steps on the 
extracted text are performed to ensure that it is in a uniform format and ready for 
comparison with the reference text. It joins all the individual transcribed sentences 
into a single string, removes any leading or trailing white spaces using the strip() 
method. Next, it removes any periods using the replace() method. Finally, it splits the 
string into individual words using the split() method and stores them in a list. The 
resulting list contains the extracted words from the Arabic speech in a uniform 
format, ready for comparison with the reference text. 

### Model:
The deep learning model built in this project is a Convolutional Neural Network 
(CNN) for audio classification. Currently the classification model words for the seven 
verses of Surah Fatiha. The model takes as input 13 MFCC coefficients (Mel
frequency cepstral coefficients) extracted from the audio clips. The model consists of 
several layers: 
- There are four sets of Convolutional and Max Pooling layers with Batch 
Normalization and ReLU activation functions to extract important features from 
the input. 
- A Flatten layer to transform the 2D feature maps obtained from the previous 
layer into a 1D feature vector. 
- Two Dense layers with ReLU activation functions and a Dropout layer to prevent 
overfitting. 
- At the end, an output Dense layer with a softmax activation function to output 
the probability distribution over the seven classes for the 7 ayats.  
The model is compiled with the ‘sparse_categorical_crossentropy’ loss function and 
‘Adam’ optimizer. It is trained with a batch size of 16 and for 25 epochs. The learning 
rate of the optimizer is set to 0.0001 for the first 25 epochs, and then reduced to 
0.000004 for another 25 epochs. Finally, the model is evaluated on a test set to 
measure its accuracy on unseen data . The trained model is saved as a .h5 file for 
future use. 
The main code executes the functions in the following order: 
- Prepare the training, validation, and test sets using prepare_datasets 
function. 
- Build a CNN model using build_model function. 
- Compile the model with an optimizer, loss function, and metrics using 
model.compile. 
- Train the model using model.fit on the training and validation sets. 
- Compile and train the model again on the training and validation sets with a 
different optimizer and learning rate. 
- Evaluate the model accuracy on the test set using model.evaluate. 
- Save the trained model to a file using model.save. 
- Load the saved model from the file using tf.keras.models.load_model. 
- Predict the class label of an audio file from the test set using predict function. 

### Detection and Correction Phase:
There are seven different verses from the Quran i.e. Surah Fatiha, represented as 
strings of Arabic text. The correctness of the recitation is being checked. It first 
extracts the words from a recited text, joining the words into a single string, and then 
splitting them into a list. It then compares the length and sequence of words of the 
extracted text to the original text. After extracting and processing the recited verses, 
the program checks whether the number of words in the recitation is the same as 
the number of words in the original verse. If they are not the same, then the 
program prints the message "Incorrect Recitation". Otherwise, the program checks 
whether the sequence of words in the recited verse is the same as in the original 
verse. If they are the same, the program prints the message "Correct Recitation". If 
the words are the same but in a different order, the program prints the message "The 
words are recited correctly, but in different order." Finally, if the recited verse 
contains incorrect words, the program prints the message "The words at index i are 
different.". This also tells what was the incorrect word along with the correct word.
## How to Run the Project

### 1. Clone the Repository
   - Clone the project repository to your local machine using the following command:
     ```
     git clone <repository_url>
     ```

### 2. Prepare Test Data
   - Record any verse from the Quran or obtain a recording from any website.
   - Convert the recorded audio into the WAV format.

### 3. Replace Test File in cnn_classifier.py
   - Open `cnn_classifier.py` in your preferred text editor or IDE.
   - Locate the function `testingOnOurVoiceAndCorrection` in `cnn_classifier.py`.
   - Replace the placeholder argument `./1/1.wav` with the path to your WAV file.
   - If your file is named differently or located in a different directory, make sure to update the path accordingly.
   - Additionally, replace the second argument `1` with the corresponding index of your WAV file.

### 4. Run the Code
   - Execute the main script or function to start the project.
   - Depending on how the project is structured, you may need to run a specific command or script to initiate the process.
   - Ensure that all dependencies are installed and configured correctly before running the code.

### 5. Review Output
   - Once the code execution is complete, review the output to see the results of the error detection and correction process.
   - The output may include information about the correctness of the recitation and any detected errors or corrections.
