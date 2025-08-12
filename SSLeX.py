"""SSLeX.py -> Stacked Super Long Exposured Images
Refactoring branch
Main user classes:

SSLeX(NameOfImages) -> Main class of Stacked Super Long eXposured images, NameOfImages is to differentiate between different SSLeX intermediate images
	Main user functions:

	AddImagesToWorkingList(PathsToImages) -> bool, Adds path to images to stack, the folder should be empty except the images you want to stack
	StackAmountOfBatches(EndingIndex: int, StartingIndex: int = 0) -> bool, Stacks StartingIndex to EndingIndex batches
"""

import math
import os
import sys
from PIL import Image
import scipy.misc
import numpy
from typing import Self
import time
import struct
import logging as lgn			#Logging for custom exceptions
from enum import Enum
import gc

sys.path.insert(0, "C:/Users/Kim Chemnitz/Documents/GitHub/Hash/")
import MCGRandom as Random

#Define some basic constants
# Sizes: list[int] = [256, 256]
Sizes: list[int] = [8160,6120]
ChannelsPerPixel: int = 3

IntermediateScalar: int = 2**20
NormalisationFactor: int = 2**10

#Initiating logging
LOGLEVEL = lgn.INFO
lgn.basicConfig(format="%(levelname)s: %(message)s", level=lgn.DEBUG)
lgn.getLogger().setLevel(LOGLEVEL)

RandomHandler = Random.MCGRandom()

def ByteToChars(Input: int) -> str:
	LowerChar = RandomHandler.ConvertableCharList[Input % 64]
	HigherChar = RandomHandler.ConvertableCharList[math.floor(Input/64)]

	return str(HigherChar + LowerChar)

def FormatSecondsToString(InputSeconds: float) -> str:
	Seconds = int(InputSeconds % 60)
	Minutes = int(math.floor(InputSeconds/60) % 60)
	Hours   = int(math.floor(InputSeconds/3600))

	return "%sH%sM%sS" % (Hours, Minutes, Seconds)

class SSLeX():
	"""SSLeX(NameOfImages) -> Main class of Stacked Super Long eXposured images, NameOfImages is to differentiate between different SSLeX intermediate images
	Main user functions:

	AddImagesToWorkingList(PathsToImages) -> bool, Adds path to images to stack, the folder should be empty except the images you want to stack
	StackAmountOfBatches(EndingIndex: int, StartingIndex: int = 0) -> bool, Stacks StartingIndex to EndingIndex batches
	"""

	def __init__(
			self: Self,
			NameOfImages: str,
		):

		self.ListOfImagePaths: list[str] = []
		self.ListOfWorkingImages: list[str] = []
		self._BaselineWorkingImagesPath_: str = ""
		self.PathToWorkingImages: str = ""
		self.PathToIntermediateImages: str = ""

		self.BatchSize: int = 8
		self.IntermediateIndex: int = 0
		self.IndexOfWorkingBatch: int = 0

		self.DoStackingMultipleBatches: bool = False
		self.DoStackingIntermediateImages: bool = False

		self.StartingIndexForBatches: int = 0
		self.EndingIndexForBatches: int = 0

		self.Name = NameOfImages

		self.WorkingImageSize: list[int] = [8160,6120]	#For 50MP RAW images
		self.BaselineChannelsPerPixel: int = 3
		self.WorkingChannelsPerPixel: int = 3

		self.LoadedImages: list[list[list[int]]] = []

		self._PercentageTracker_: int = 0
		self._LastTime_: float = 0
		self._LastDelta_: float = 0

		self.VerboseDebugging: bool = False
	
	def AddImagesToWorkingList(
			self: Self, 
			PathsToImages: str
		) -> bool:

		self.ListOfImagePathsFiles = os.listdir(PathsToImages)
		self.PathToWorkingImages = PathsToImages

		return True
	
	def LoadImages(
			self: Self,
		) -> bool:

		CheckingImage = self.ListOfWorkingImages[0].split(".")[1].lower()

		if CheckingImage == "tif" or CheckingImage == "png" or CheckingImage == "tiff":
			for ImageToLoad in self.ListOfWorkingImages:
				with Image.open(self.PathToWorkingImages + ImageToLoad) as LoadedImage:
					self.LoadedImages.append(list(LoadedImage.getdata()))
		elif CheckingImage == "intermediate":
			for ImageToLoad in self.ListOfWorkingImages:
				DidLoadImage = self.LoadIntermediateImage(self.PathToWorkingImages, ImageToLoad, self.WorkingImageSize, self.WorkingChannelsPerPixel)
				if not DidLoadImage:
					raise Exception
		else:
			raise TypeError("Imagetype %s is not supported." % (CheckingImage))
		
		return True

	def GetDimensions(
			self: Self,
			ImagePath: str
		) -> bool:

		with Image.open(self.PathToWorkingImages + ImagePath) as WorkingImage:
			self.WorkingImageSize = WorkingImage.size
		
		return True

	def SaveIntermediateImage(
			self: Self,
			IntermediateImage: "NDArray",
		) -> bool:
		
		lgn.info("Saving Intermediate image, batch number %s" % (self.IndexOfWorkingBatch))

		IntermediateImages = [Image.new("F", self.WorkingImageSize) for _ in range(self.WorkingChannelsPerPixel)]

		if self.DoStackingIntermediateImages:
			IntermediateImages[0].putdata(IntermediateImage.flatten())
		else:
			for ChannelIndex in range(self.WorkingChannelsPerPixel):
				IntermediateImages[ChannelIndex].putdata(IntermediateImage[ChannelIndex].flatten())

		#Save intermediate images in seperate images for full float precision
		lgn.debug("Splitting image to each channel for ssaving in full quality.")

		if self.DoStackingIntermediateImages:
			for ChannelIndex in range(self.WorkingChannelsPerPixel):
				IntermediateImages[ChannelIndex].save(self.PathToIntermediateImages + "Intermediate%sChannel/%sBatch%s.tiff" % (self.IntermediateIndex, self.Name, self.IndexOfWorkingBatch))
		else:
			for ChannelIndex in range(self.WorkingChannelsPerPixel):
				IntermediateImages[ChannelIndex].save(self.PathToIntermediateImages + "Intermediate%sChannel/%sBatch%s.tiff" % (ChannelIndex, self.Name, self.IndexOfWorkingBatch))

		lgn.info("Saved to disk")

		return True

	def LoadIntermediateImage(
			self: Self,
			PathToIntermediateImage: str,
			NameOfImage: str,
			Sizes: list[int],
			ChannelsPerPixel: int,
		) -> bool:
		
		FileHandler = open(PathToIntermediateImage + NameOfImage)
		FileRead = FileHandler.read()

		Values = FileRead.split(":")
		Values.pop(-1)

		TemporaryImage = Image.new("RGB", Sizes)

		ValueTracker = 0
		for x in range(Sizes[0]):
			for y in range(Sizes[1]):
				TemporaryPixel: list[int] = []
				for ChannelIndex in range(ChannelsPerPixel):
					TemporaryPixel.append(int(int(Values[ValueTracker])/IntermediateScalar))
					ValueTracker += 1
				TemporaryImage.putpixel([x,y], tuple(TemporaryPixel))

		TemporaryImage.show()
		TemporaryImage.save(PathToIntermediateImage + "Image.png")

		return True

	def StackBatchOfImages(
			self: Self,
			PathToIntermediateImages: str
		) -> bool:

		#Collect garbage
		gc.collect()
		self._PercentageTracker_ = 0

		self.PathToIntermediateImages = PathToIntermediateImages

		lgn.info("Stacking batch %s" % (self.IndexOfWorkingBatch))

		lgn.debug("Loading images to process to memory")

		lgn.debug("ListOfImagePathsFiles: %s" % (self.ListOfImagePathsFiles))

		self.ListOfWorkingImages = []
		#Prepare list of images to stack
		for i in range(self.BatchSize):
			IndexOfImage: int = i + self.IndexOfWorkingBatch * self.BatchSize

			self.ListOfWorkingImages.append(
				self.ListOfImagePathsFiles[IndexOfImage]
			)
		
		# Backup GetDimensions in case dimensions have changed
		self.GetDimensions(self.ListOfWorkingImages[0])
		
		DidLoadedImages: bool = self.LoadImages()
		if not DidLoadedImages:
			lgn.error("Did not load imaged correctly.")
			return False
		
		self._LastTime_ = time.time()
		self._LastDelta_ = 0

		MaxPixelIndex = self.WorkingImageSize[0] * self.WorkingImageSize[1]

		lgn.debug("Entering Main Stacking Loop <3")

		if self.DoStackingIntermediateImages:
			IntermediateImageArray = numpy.zeros(shape=[self.WorkingImageSize[0], self.WorkingImageSize[1]], dtype=numpy.float64)
			lgn.debug("Created Intermediate Image Array.")

			for x in range(self.WorkingImageSize[0]):
				for y in range(self.WorkingImageSize[1]):
					PixelIndex: int = self.WorkingImageSize[1]*x+y
					for ImageIndex in range(self.BatchSize):
						#Gamma corrected average of images m.sqrt((Imaged1 ** 2 + Image2 ** 2)/2)
						IntermediateImageArray[tuple([x, y])] += (self.LoadedImages[ImageIndex][PixelIndex] ** 2)/self.BatchSize
					self.PrintPercentage(PixelIndex, MaxPixelIndex)
		else:
			IntermediateImageArray = numpy.zeros(shape=[self.WorkingChannelsPerPixel, self.WorkingImageSize[0], self.WorkingImageSize[1]], dtype=numpy.float64)
			lgn.debug("Created Intermediate Image Array.")

			for x in range(self.WorkingImageSize[0]):
				for y in range(self.WorkingImageSize[1]):
					PixelIndex: int = self.WorkingImageSize[1]*x+y
					for ImageIndex in range(self.BatchSize):
						for ChannelIndex in range(self.WorkingChannelsPerPixel):
							#Gamma corrected average of images m.sqrt((Imaged1 ** 2 + Image2 ** 2)/2)
							IntermediateImageArray[tuple([ChannelIndex, x, y])] += (self.LoadedImages[ImageIndex][PixelIndex][ChannelIndex] ** 2)/self.BatchSize
					self.PrintPercentage(PixelIndex, MaxPixelIndex)

		#Correct for gamma
		lgn.debug("Correcting for gamma")
		GammaCorrectedArray = numpy.sqrt(IntermediateImageArray)

		#Normalising image, because dng images values goes from 0 to 2**10, whilst tiff requires from 0 to 1
		if not self.DoStackingIntermediateImages:
			lgn.debug("Normalising image")
			for x in range(self.WorkingImageSize[0]):
				for y in range(self.WorkingImageSize[1]):
					for ChannelIndex in range(self.WorkingChannelsPerPixel):
						GammaCorrectedArray[tuple([ChannelIndex, x, y])] = GammaCorrectedArray[tuple([ChannelIndex, x, y])]/NormalisationFactor

		lgn.debug("Stacked, now saving intermediate image")
		SavedImage: bool = self.SaveIntermediateImage(GammaCorrectedArray)
		if not SavedImage:
			return False
		
		lgn.debug("Intermediate image saved.")

		lgn.debug("Deleting LoadedImages")
		self.LoadedImages = []
		lgn.debug("Images deleted.")

		return True
	
	def PrintPercentage(
			self: Self,
			IndexDone: int,
			MaxIndex: int
		) -> None:

		#Make time estimation cleaner
		if 100*IndexDone/MaxIndex >= self._PercentageTracker_:
			NowTime = time.time()
			NowDeltaTime: float = NowTime - self._LastTime_
			CleansedDelta: float = (NowDeltaTime + self._LastDelta_)/2

			ThisBatchesEstimatedTime: float = CleansedDelta * (100-self._PercentageTracker_ )
			FormattedEstimatedTime: str = FormatSecondsToString(ThisBatchesEstimatedTime)
			
			if self._PercentageTracker_  != 0:
				if self.DoStackingMultipleBatches:
					EstimatedTimeForRestOfBatches: float = 100*CleansedDelta*(self.EndingIndexForBatches-self.IndexOfWorkingBatch)
					OverallEstimatedTime: float = ThisBatchesEstimatedTime + EstimatedTimeForRestOfBatches
					FormattedEstimatedAllBatchesTime: str = FormatSecondsToString(OverallEstimatedTime)
					lgn.info("%s%% done for this batch, estimated time: %s" % (self._PercentageTracker_ , FormattedEstimatedAllBatchesTime))
				else:
					lgn.info("%s%% done, estimated time for this batch: %s" % (self._PercentageTracker_ , FormattedEstimatedTime))
	
			self._LastTime_ = NowTime
			self._LastDelta_ = NowDeltaTime
			self._PercentageTracker_ += 1

	def StackAmountOfBatches(
			self: Self, 
			PathToSourceImages: str,
			PathToIntermediateImages: str,
			EndingIndex: int, 
			StartingIndex: int = 0
		) -> bool:

		gc.collect()

		self.PathToWorkingImages = PathToSourceImages
		self.PathToIntermediateImages = PathToIntermediateImages

		self.DoStackingMultipleBatches = True
		self.StartingIndexForBatches = StartingIndex
		self.EndingIndexForBatches = EndingIndex

		DidAddImages: bool = self.AddImagesToWorkingList(self.PathToWorkingImages)
		if not DidAddImages:
			raise Exception

		lgn.info("Stacking batch %s to %s" % (StartingIndex, EndingIndex))

		for i in range(StartingIndex, EndingIndex):
			self.IndexOfWorkingBatch = i
			DidStackingOnBatch: bool = self.StackBatchOfImages(self.PathToIntermediateImages)
			if not DidStackingOnBatch:
				raise Exception

		lgn.info("Finished stacking batches of images.")

		return True
	
	def StackIntermediateImages(
			self: Self,
			PathToSourceImages: str,
			PathToIntermediateImages: str,
			EndingIndex: int,
			StartingIndex: int = 0
		) -> bool:
		
		self._BaselineWorkingImagesPath_ = PathToSourceImages
		self.DoStackingIntermediateImages = True

		self.BaselineChannelsPerPixel = 3
		self.WorkingChannelsPerPixel = 1
		for ChannelIndex in range(self.BaselineChannelsPerPixel):
			lgn.debug("Stacking Channel %s" % (ChannelIndex))
			self.IntermediateIndex = ChannelIndex
			self.PathToWorkingImages = self._BaselineWorkingImagesPath_ + "Intermediate%sChannel" % (ChannelIndex)
			DidStackingThisChannel: bool = self.StackAmountOfBatches(
				PathToSourceImages + "Intermediate%sChannel/" % (ChannelIndex),
				PathToIntermediateImages,
				EndingIndex,
				StartingIndex
			)

			if not DidStackingThisChannel:
				raise Exception

		return True

def CreateRandomisedImage(
		OutputPath: str,
		OutputName: str,
		IndexOfImage: int,
	):

	global Seed

	LocalRandomHandler = Random.MCGRandom()

	Sizes: list[int] = [256,256]
	ChannelsPerPixes: int = 3
	MaxSubPixelValue: int = 2**8
	MaxPixelIndex: int = Sizes[0] * Sizes[1]

	TemporaryImage = numpy.zeros([Sizes[1], Sizes[0], ChannelsPerPixel], dtype=numpy.uint64)
	for x in range(Sizes[0]):
		for y in range(Sizes[1]):
			for ChannelIndex in range(ChannelsPerPixes):
				# RandomValue: int = RandomHandler.RandomInt(None, 0, MaxSubPixelValue)

				PixelIndex = Sizes[1]*y + x
				PixelPercentage = PixelIndex/MaxPixelIndex

				TemporaryImage[x][y][ChannelIndex] = math.floor(256*PixelPercentage*IndexOfImage/8)

	OutputImage = Image.new("RGB", Sizes)

	for x in range(Sizes[0]):
		for y in range(Sizes[1]):
			TemporaryPixel = []
			for ChannelIndex in range(ChannelsPerPixel):
				TemporaryPixel.append(TemporaryImage[x][y][ChannelIndex])
			OutputImage.putpixel([x,y], tuple(TemporaryPixel))

	OutputImage.save(OutputPath + OutputName + ".png")

class FunctionsToCall(Enum):
	DoNothing = 0
	CreateRandomisedImages = 1
	StackActualImages = 2
	StackTestImages = 3
	StackIntermediateImages = 4
	LoadImages = 5

FunctionToCall: "FunctionsToCall" = FunctionsToCall.StackIntermediateImages

def __main__(vArgs: list[str] = []) -> bool:
	global FunctionToCall
	SSLeXHandler: "SSLeX" = SSLeX(vArgs[0])

	ReturnValue: bool
	if FunctionToCall == FunctionsToCall.DoNothing:
		ReturnValue= DoNothing(vArgs)
	elif FunctionToCall == FunctionsToCall.CreateRandomisedImages:
		ReturnValue = CreateRandomisedImages(vArgs)
	elif FunctionToCall == FunctionsToCall.StackActualImages:
		ReturnValue = StackActualImages(vArgs)
	elif FunctionToCall == FunctionsToCall.StackTestImages:
		ReturnValue = StackTestImages(vArgs)
		input()
	elif FunctionToCall == FunctionsToCall.StackIntermediateImages:
		ReturnValue = StackIntermediateImages(vArgs)
	elif FunctionToCall == FunctionsToCall.LoadImages:
		ReturnValue = LoadImages(vArgs)
	else:
		raise ValueError("Incorrect value of FunctionToCall.")
	
	return ReturnValue

#Load image
def LoadImages(vArgs: list[str] = []) -> bool:
	SSLeX.LoadIntermediateImage(SSLeX, "D:/Users/hatel/Pictures/SSLeXIntermediate/", "Intermediate.intermediate", Sizes, ChannelsPerPixel)

	return True

def DoNothing(vArgs: list[str]) -> bool:
	return True

#Create randomised images
def CreateRandomisedImages(vArgs: list[str] = []) -> bool:
	lgn.debug("Creating Randomised Images")
	
	for i in range(8):
		CreateRandomisedImage("D:/Users/hatel/Pictures/SSLeXTest/", str(i), i)
		lgn.debug("Created %sth image" % (i))
	
	return True

#Normal stack
def StackActualImages(vArgs: list[str] = []) -> bool:
	if len(vArgs) < 4:
		raise AttributeError("vArgs: list[str] requires [Name], [Path to source images], [Path to intermediate images], [Ending index], [Starting index]")

	SSLeXHandler: "SSLeX" = SSLeX(vArgs[0])

	try:
		StartingIndex: int = int(vArgs[4])
	except IndexError:
		StartingIndex = 0

	DidStack: bool = SSLeXHandler.StackAmountOfBatches(vArgs[1], vArgs[2], int(vArgs[3]), int(StartingIndex))

	if not DidStack:
		return False
	
	return True

def StackIntermediateImages(vArgs: list[str] = []) -> bool:
	if len(vArgs) < 4:
		raise AttributeError("vArgs: list[str] requires [Name], [Path to source images], [Path to intermediate images], [Ending index], [Starting index]")

	SSLeXHandler: "SSLeX" = SSLeX(vArgs[0])

	try:
		StartingIndex: int = int(vArgs[4])
	except IndexError:
		StartingIndex = 0
	
	SSLeXHandler.BatchSize = int(vArgs[5])

	for ChannelIndex in range(SSLeXHandler.WorkingChannelsPerPixel):
		DidStack: bool = SSLeXHandler.StackIntermediateImages(vArgs[1], vArgs[2], int(vArgs[3]), int(StartingIndex))

	if not DidStack:
		return False
	
	return True

def StackTestImages(vArgs: list[str] = []) -> bool:
	global NormalisationFactor
	NormalisationFactor = 256

	SSLeXHandler: "SSLeX" = SSLeX("TestImages")

	SSLeXHandler.AddImagesToWorkingList("D:/Users/hatel/Pictures/SSLeXTest/")

	DidStack: bool = SSLeXHandler.StackBatchOfImages("D:/Users/hatel/Pictures/SSLeXIntermediate/")

	if not DidStack:
		return False
	
	return True
