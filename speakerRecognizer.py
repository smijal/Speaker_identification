import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
import warnings
warnings.filterwarnings("ignore")
import os
import time
import pyaudio
import wave
import glob
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

#Function to record audio for given duration
def recordAudio(duration):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    RATE = 16000
    RECORD_SECONDS = duration

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=2,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return p,frames

#To save the recorded audio as wave file
def saveAudio(p,frames, path):
	try:
		wf = wave.open(path, 'wb')
		wf.setnchannels(2)
		wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
		wf.setframerate(16000)
		wf.writeframes(b''.join(frames))
		wf.close()
	except:
		print("There was a problem with the audio file...")

############################################################

#Derivatives od MFCC's calculation
# This function is taken from internet to help the accuracy of my model
def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < rows-1:
                second = rows-1
            else:
                second = i+j
            index.append((second,i))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


# Extracts features from audio with given rate,
# Gets 20 mfcc features and calculates additional 20 derivatives
def extract_features(audio,rate): 
    mfcc_feat = mfcc.mfcc(audio,rate, winlen=0.025, winstep=0.01,numcep=20,appendEnergy = True)
    mfcc_feat = preprocessing.scale(mfcc_feat) #scale the features
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta)) # combines mfcc's with calculated derivatives in one numpy array
    return combined


############## The following 2 functions are meant for getting the pitch value, 
# but I am not using it since it is not accurate
def average(x):
	avg = 30
	for i in range(len(x)-1):
		j=i+1
		diff = x[j]-x[i]
		avg = (avg+diff)/2
	return avg
def estimatePitch(x):
	shape = x.shape
	print(shape)
	if(len(shape) == 1):
		x = x[0:10000].reshape(10000)
	else:
		x = x[0:10000,0].reshape(10000)
	result = np.correlate(x, x, mode='full')
	size=int(result.shape[0])
	result = result[size//2:size//2+1000]
	print("max ", np.max(result))
	peaks, _ = find_peaks(result, height=np.max(result)-200)
	plt.plot(result)
	plt.plot(peaks, result[peaks], "x")
	plt.plot(np.zeros_like(result), "--", color="gray")
	plt.show()
	return average(peaks)

#######################################################################
#### To create directory for different people and models
def createDirectory(main_path,directory_name):
	directory = os.path.join(main_path,directory_name)
	if not os.path.exists(directory):
		os.makedirs(directory)
		print("Directory ", directory_name, "successfully created.")
		return directory
	else:
		print("Directory name already exists...")
		return None
def createPermanent(main_path,directory_name):
	directory = os.path.join(main_path,directory_name)
	if not os.path.exists(directory):
		os.makedirs(directory)
	else:
		print("Directory name already exists")
	return directory


#To fit a GMM model and save it as a pickle file
def createModel(audio_path,models_path, name,num_files):
	features = np.asarray(())
	a_path=os.path.join(audio_path,"*")
	count = 1
	for path in glob.glob(a_path): 
		print(path)
		sr,audio = read(path)
		# extract 40 dimensional MFCC & delta MFCC features
		#print(estimatePitch(audio))
		vector   = extract_features(audio,sr)
		if features.size == 0:
			features = vector
		else:
			features = np.vstack((features, vector))
		# when features of 5 files of speaker are concatenated, then do model training
		if count == num_files:
			gmm = GMM(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
			gmm.fit(features)
			# dumping the trained gaussian model
			picklefile = os.path.join(models_path,name+".gmm")
			cPickle.dump(gmm,open(picklefile,'wb'))
			print ('+ Modeling completed for speaker:',name," with data point = ",features.shape)
			features = np.asarray(())
			count = 0
		count = count + 1

# To identify the person
def identifyPerson(models_path, folder):
	models_path = os.path.join(models_path, "*")
	gmm_files = glob.glob(models_path)
	models = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
	if(len(models)<=1):
		print("Database is empty.")
		return None
	speakers   = [fname.split("\\")[-1].split(".gmm")[0].replace(models_path,"") for fname in gmm_files]
	print("Speakers found in database: ", speakers)
	folder = os.path.join(folder,"currentSpeaker.wav")
	#RECORDING PERSON FOR IDENTIFICATION
	inp = input("Press ENTER to record a 10sec long audio...")
	p,frames = recordAudio(10)
	saveAudio(p,frames,folder)
	sr,audio = read(folder)
	vector   = extract_features(audio,sr)
	log_likelihood = np.zeros(len(models)) 

	for i in range(len(models)):
		gmm    = models[i]  #checking with each model one by one
		scores = np.array(gmm.score(vector))
		log_likelihood[i] = scores.sum()

	winner = np.argmax(log_likelihood)

	print ("\tDetected as - ", speakers[winner])
	print("\tLog likelihoods: "+ str(log_likelihood))


def main():
	print("###################################################################")
	print("###		GMM-based speaker identification system		###")
	print("###################################################################")
	folder =  os.path.dirname(os.path.abspath(__file__))
	#path to training data
	audio_database = createPermanent(folder,"audio_database")
	#path where trained speaker models will be saved
	models = createPermanent(folder,"gmm_models")
	garbage_model = os.path.join(folder,"garbage")
	createModel(garbage_model, models, "garbage",10)
	yes_no = {"Y":True, "N":False}

	while(True):
		print("---------------------------------------------------------")
		response = input("\n1. Add a new person to the database\n2. Identify person\n3. Exit ")
		if(response=='1'):
			print("Training")
			name = input("Enter the new person's name: ").lower()
			person_path = createDirectory(audio_database, name)
			if(person_path==None):
				overwrite=yes_no[input("Do you want to overwrite it? [Y/N]")[0].upper()]
				if(not overwrite):
					continue
				else:
					person_path = createPermanent(audio_database,name)
			print("In order to train a new model, " + str(name) + " is required to speak for 50 seconds.")
			print("We recommend reading random paragraphs.")
			print("There will be 5 recordings, 10 seconds each.")
			for i in range(5):
				a_path = os.path.join(person_path,name+str(i)+".wav")
				inp = input("Press ENTER to record a 10sec long audio...")
				p,frames = recordAudio(10)
				saveAudio(p,frames,a_path)
			print("Audio recordings successfully saved.")
			createModel(person_path,models,name,5)
		elif(response=='2'):
			identifyPerson(models,folder)
			continue
		elif(response=='3'):
			exit()
		else:
			print("Error... Enter a valid choice [1-2]")

if __name__ == "__main__":
	main()
