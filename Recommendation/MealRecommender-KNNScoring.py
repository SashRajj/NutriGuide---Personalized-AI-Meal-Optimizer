import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("user_Profiles.csv")
userData = data.values

data = pd.read_csv("dataset.csv")
mealData = data.values

K = 500
K_List = [0]*K

userRows = 98
userCols = 5

userInput = [[0]*userCols for n in range(userRows)]

mealRows = 512
mealCols = 9
planCols = 5
mealPlan = [[0]*planCols for n in range(mealRows)]


userPoints = [[0] for n in range(userRows)]
mealPoints = [[0] for n in range(mealRows)]
error = [[0]*3 for n in range(mealRows)]

temp = 0
tmpCount = 0

maxCount = 0
maxIndex = 0

accuracy = 0
cumAccuracy = [0]*(K+1)
totalAccuracy = [0]*(K+1)
outputAccuracy = [0]*K

flag = 0
threshold = 0


# Initialize K_List

for n in range(0, K):
	K_List[n] = n+1


# Extract all of the user input profile data (to be used as the test data for this KNN Algorithm)
# Also compute the total point score for each user input profile

for i in range(userRows):
	for j in range(userCols):
		userInput[i][j] = userData[i][j]

		if (j != 0):
			tmpCount += len(userInput[i][j].split())

	userInput[i][0] = i+1
	userPoints[i] = tmpCount
	tmpCount = 0

maxPoints = max(userPoints)
errorCounts = [0]*maxPoints


# Extract all of the meal input data (to be used as the training data for this KNN Algorithm)

for i in range(mealRows):
	for j in range(mealCols):
		if (j == 0):
			mealPlan[i][j] = mealData[i][j]

		if (j >= 4 and j <= 7):
			mealPlan[i][j-3] = mealData[i][j]

	mealPlan[i][0] = i+1



# Begin KNN Algorithm to find the best match meal plans for each user input profile (for different K nearest neighbors values)
# Also compute the prediction accuracy for each different K nearest neighbor value and plot the results at the end of this program

for k in range(1, K+1):
	for i in range(0, userRows):
		for j in range(0, mealRows):
			for n in range(1, userCols):
				temp = userInput[i][n].split()

				# Compute the meal plan score and determine how many points off it is from the user input profile score
				for p in range(0, len(temp)):
					if (mealPlan[j][n].find(temp[p]) != -1):
						tmpCount += 1

			mealPoints[j] = tmpCount
			tmpCount = 0

			error[j][0] = abs(userPoints[i] - mealPoints[j])
			error[j][1] = i
			error[j][2] = j

		# Sort all of the scores by minimum error margin between meal plan and the current user input profile

		error.sort()

		# Compute the polling to determine which error margin is most frequent among the top K nearest neighbors

		for n in range(0, k):
			errorCounts[error[n][0]] += 1

		maxCount = np.max(errorCounts)
		maxIndex = np.where(errorCounts == maxCount)

		# Compute the accuracy based on the most frequent error margin among the top K nearest neighbors (using a percentage score = meal points / user points)
		# Then add this accuracy to the cumulative accuracy computed so far for each user input profile

		accuracy = (userPoints[i] - maxIndex[0][0]) / float(userPoints[i])
		cumAccuracy[k] += accuracy

		for n in range(0, maxPoints):
			errorCounts[n] = 0


	totalAccuracy[k] = (cumAccuracy[k] / float(userRows)) * 100

	if (totalAccuracy[k] <= 50 and flag == 0):
		flag = 1
		threshold = k
	

# Plot the totalAccuracy output per K nearest neighbor value in a graph (over the given range of input K values provided)
# Convert totalAccuracy to an output format suitable for graphing (to clean up the shift and remove the extraneous zero at the header of the list)

for n in range(0, K):
	outputAccuracy[n] = totalAccuracy[n+1]
	
plt.plot(K_List, outputAccuracy)
plt.xlabel('List of K Nearest Neighbor Values')
plt.ylabel('Meal Prediction Accuracy  (in percentage format)')
plt.xlim(1, 500)
plt.ylim(0, 100)
plt.show()


print(totalAccuracy[threshold], threshold)
print(totalAccuracy[K], K)
print("\n")
print("Finished!")


