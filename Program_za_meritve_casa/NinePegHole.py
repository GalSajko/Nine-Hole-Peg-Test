import numpy as np 
import cv2 as cv
import time

cap = cv.VideoCapture(1)
runningFrameCounter = 0
setupFrameCounter = 0
maxFrame = 3
settedPinsCounter = 0
confirmedPins = 0
lastConfirmedPins = 0
borderHolesCoords = []
platformEdges = []
yArea = [0, 0]
measuredTimes = []
allPinsSetted = False
allPinsDown = False
start = False
nextTry = False
handOnRight = False
handOnLeft = False

def calculateThreshold(grayImage):
	tO = cv.threshold(grayImage, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[0]
	tL = tO / 2
	tH = tO

	return tL, tH

def gammaImage(grayImage, gamma):
	dtype = grayImage.dtype
	grayImage = grayImage.astype('float')
	if dtype.kind in ('u', 'i'):
		minValue = np.iinfo(dtype).min
		maxValue = np.iinfo(dtype).max
	else:
		minValue = np.min(grayImage)
		maxValue = np.max(grayImage)
	rangeValue = maxValue - minValue

	grayImage = (grayImage - minValue) / float(rangeValue)
	gammaImg = grayImage**gamma
	gammaImg = float(rangeValue) * gammaImg + minValue

	gammaImg[gammaImg < 0] = 0
	gammaImg[gammaImg >255] = 255

	return gammaImg.astype(dtype)

def rgbToHsv(rgbImage):
	r = rgbImage[:, :, 2]
	g = rgbImage[:, :, 1]
	b = rgbImage[:, :, 0]

	cMax = np.maximum(r, np.maximum(g, b))
	cMin = np.minimum(r, np.minimum(g, b))
	delta = cMax - cMin + 1e-7

	h = np.zeros_like(r)
	s = np.zeros_like(r)
	v = np.zeros_like(r)

	h[cMax == r] = 60.0 * ((g[cMax == r] - b[cMax == r]) / (delta[cMax == r]) % 6.0)
	h[cMax == g] = 60.0 * ((b[cMax == g] - r[cMax == g]) / (delta[cMax == g]) + 2)
	h[cMax == b] = 60.0 * ((r[cMax == b] - g[cMax == b]) / (delta[cMax == b]) + 4)

	s[delta != 0] = delta[delta != 0] / (cMax[delta != 0] + 1e-7)

	v = cMax

	return h, s, v

def prepareImage(frame):
	# Gray image of frame.
	grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	# HSV space of (colored) frame.
	h, s, v = rgbToHsv(frame / 255)
	# Gamma image to improve contrast on gray image.
	gammaImg = gammaImage(grayImg, 1.5)
	# Blurred gamma image.
	blurImg = cv.GaussianBlur(gammaImg, (7, 7), sigmaX = 1.5, sigmaY = 1.5)
	# Automatic thresholding on blur image.
	tH = calculateThreshold(blurImg)[1]

	return blurImg, tH, h, s, v

def findCircles(blurImg, tH):
	# Find candidates for setted pins.
	pinsCandidates = cv.HoughCircles(blurImg, cv.HOUGH_GRADIENT, 3, 50, param1 = tH, param2 = 2, minRadius = 3, maxRadius = 10)
	# Find collection area for unsetted pins.
	collectionArea = cv.HoughCircles(blurImg, cv.HOUGH_GRADIENT, 3, 50, param1 = tH, param2 = 100, minRadius = 100, maxRadius = 150)	

	return pinsCandidates, collectionArea

def detectHand(whitePixelsNumber):
	handState = True if whitePixelsNumber > 1000 else False
	return handState


def setBorderHolesPosition():
	input("Put all pins in collection area and press any key to continue.")
	print("Click on upper left and lower right hole, than click on upper left and lower right edge of platform. Than press ENTER")
	global borderHolesCoords
	borderHolesCoords = []
	while True:
		ret, frame = cap.read()
		cv.setMouseCallback('frame', onMouse)
		for (x, y) in borderHolesCoords:
			cv.circle(frame, (x, y), 3, (0, 0, 255), 5)
		cv.imshow('frame', frame)
		if cv.waitKey(1) & 0xFF == 13 and len(borderHolesCoords) >= 4:
			break
	print("READY!")

def onMouse(event, x, y, flags, params):
	global borderHolesCoords
	ret, frame = cap.read()
	if event == cv.EVENT_LBUTTONDOWN:
		borderHolesCoords.append((x, y))

		
if __name__ == '__main__':
	# Find holes area.
	setBorderHolesPosition()
	lastFrame = 0
	lastFrameGray = 0
	openingKernel = np.ones((9, 9), np.uint8)
	handX, handY = 0, 0
	captureFrameCounter = 1
	radius = 0
	pinTimes = [0]
	pinTimesDiff = []
	numberOfTries = 0
	lastPins, pins = 0, 0
	# Cammera running loop.
	while(True):
		if (nextTry):
			setBorderHolesPosition()
			lastFrameGray = 0
			pinTimes = [0]
			pinTimesDiff = []
			nextTry = False
		# Original frame from video live feed.
		setupFrameCounter += 1
		ret, frame = cap.read()
		# Use only part of the image with platform on it.
		platformFrame = frame[borderHolesCoords[2][1]:borderHolesCoords[3][1], borderHolesCoords[2][0]:borderHolesCoords[3][0]]
		platformFrameGray = cv.cvtColor(platformFrame, cv.COLOR_BGR2GRAY)

		# Compute difference between two consecutive frames and use opening to remove noise.
		currentFrameGray = platformFrameGray
		platformDiffGray = cv.morphologyEx(currentFrameGray - lastFrameGray, cv.MORPH_OPEN, openingKernel)
		lastFrameGray = currentFrameGray

		numberOfWhitePixels = 0
		if (setupFrameCounter > 1):
			numberOfWhitePixels = len(np.where(platformDiffGray > 200)[0])

		handInImage = detectHand(numberOfWhitePixels)

		if (handInImage and (start == False)):
			print("TIME STARTED")
			startTime = time.time()	
			pinTimes[0] = startTime		
			start = True

		# Start moving pins.
		if (start):
			blurImg, tH, h, s, v = prepareImage(frame)
			pinsCandidates, collectionArea = findCircles(blurImg, tH)

			runningFrameCounter += 1
			handInImage = detectHand(numberOfWhitePixels)

			# Find position of hand on every 2nd frame.
			if (runningFrameCounter % 2 == 0):
				radius = 0
				if (handInImage):
					radius = 3
					handX = np.mean(np.where(platformDiffGray > 200)[0])
					handY = np.mean(np.where(platformDiffGray > 200)[1])					

			# Draw red circle on the middle of the hand.		
			cv.circle(frame, (int(handY + borderHolesCoords[2][0]), int(handX + borderHolesCoords[2][1])), radius, (0, 0, 255), 5)


			if (numberOfTries == 0):
				handCondition = handY + borderHolesCoords[2][0] > borderHolesCoords[1][0] + 20
			else:
				handCondition = handY + borderHolesCoords[2][0] < borderHolesCoords[0][0] - 20

			nonConfirmedPins = 0
			if (handCondition):
				# Find actual setted pins with green circles on top.
				if pinsCandidates is not None:
					pinsCandidates = np.floor(pinsCandidates[0, :]).astype('int')
					for (x, y, r) in pinsCandidates:
						# Look only in holes area.
						if (x > borderHolesCoords[0][0] - 20 and x < borderHolesCoords[1][0] + 20 and y > borderHolesCoords[0][1] - 20 and y < borderHolesCoords[1][1] + 20):
							# Look in hsv space for green circle.
							if (h[y, x] < 190):
								settedPinsCounter += 1
								nonConfirmedPins += 1
								cv.circle(frame, (x, y), r, (255, 255, 255), 1)
			
			# Count setted pins on each *maxFrame* frame.
			lastConfirmedPins = confirmedPins
			if (nonConfirmedPins < 9):
				if (runningFrameCounter % maxFrame == 0):
					confirmedPins = np.round(settedPinsCounter / maxFrame).astype('int')
					#print(confirmedPins)
					settedPinsCounter = 0
			elif (nonConfirmedPins == 9):
				confirmedPins = nonConfirmedPins
				settedPinsCounter = 0
				#print(confirmedPins)

			if (confirmedPins > lastConfirmedPins and not allPinsSetted):
				pinTime = time.time()
				pinTimes.append(pinTime)
				singlePinTime = round(pinTimes[-1] - pinTimes[-2], 2)
				if (singlePinTime > 0.9):
					print("Pin time ", singlePinTime)
					pinTimesDiff.append(singlePinTime)

			# Set flag if all pins are setted.
			if (confirmedPins == 9):
				allPinsSetted = True
				print("ALL PINS SETTED")
			# Set flag if all pins are down.
			if (allPinsSetted and (confirmedPins == 0)):
				allPinsDown = True
			# Break loop after all pins are down (if they were setted before) and hand is not in the image.
			if (not handInImage and allPinsDown):
				endTime = time.time()
				elapsedTime = round(endTime - startTime, 2)
				measuredTimes.append(elapsedTime)
				if (len(measuredTimes) > 1):
					print("Difference between first and second hand: " + str(round(measuredTimes[0] - measuredTimes[1], 2)))
					measuredTimes = []
				allPinsSetted = False
				allPinsDown = False
				setupFrameCounter = 0
				runningFrameCounter = 0
				print("FINISHED IN ", elapsedTime, "SECONDS")
				print("Fastest pin: ", np.min(pinTimesDiff))
				print("Slowest pin: ", np.max(pinTimesDiff))
				print("Average time per pin: ", np.mean(pinTimesDiff))
				print("Number of timestamps: ", str(len(pinTimesDiff)))
				start = False
				if (numberOfTries == 0):
					input("Press ENTER to repeat.")
					nextTry = True
					numberOfTries += 1
				else:
					break

		cv.imshow('frame', frame)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv.destroyAllWindows()

