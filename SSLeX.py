"""SSLeX.py -> Stacked Super Long Exposured Images
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
		self.PathToWorkingImages: str = ""
		self.PathToIntermediateImages: str = ""

		self.BatchSize: int = 8
		self.IndexOfWorkingBatch: int = 0

		self.DoStackingMultipleBatches: bool = False
		self.StartingIndexForBatches: int = 0
		self.EndingIndexForBatches: int = 0

		self.Name = NameOfImages

		self.WorkingImageSize: list[int] = [8160,6120]	#For 50MP RAW images
		self.ChannelsPerPixel: int = 3

		self.LoadedImages: list[list[list[int]]] = []
	
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

		if CheckingImage == "tif" or CheckingImage == "png":
			for ImageToLoad in self.ListOfWorkingImages:
				with Image.open(self.PathToWorkingImages + ImageToLoad) as LoadedImage:
					self.LoadedImages.append(list(LoadedImage.getdata()))
		elif CheckingImage == "intermediate":
			for ImageToLoad in self.ListOfWorkingImages:
				DidLoadImage = self.LoadIntermediateImage(self.PathToWorkingImages, ImageToLoad, self.WorkingImageSize, self.ChannelsPerPixel)
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
			IntermediateImageChannel0: "NDArray",
			IntermediateImageChannel1: "NDArray",
			IntermediateImageChannel2: "NDArray",
		) -> bool:
		
		lgn.info("Saving Intermediate image, batch number %s" % (self.IndexOfWorkingBatch))

		RedChannelIntermediateImage = Image.new("F", self.WorkingImageSize)
		GreenChannelIntermediateImage = Image.new("F", self.WorkingImageSize)
		BlueChannelIntermediateImage = Image.new("F", self.WorkingImageSize)

		RedChannelIntermediateImage.putdata(IntermediateImageChannel0.flatten())
		GreenChannelIntermediateImage.putdata(IntermediateImageChannel1.flatten())
		BlueChannelIntermediateImage.putdata(IntermediateImageChannel2.flatten())

		#Save intermediate images in seperate images for full float precision
		lgn.debug("Splitting image to each channel for ssaving in full quality.")

		lgn.debug("Saving:")
		lgn.debug(self.PathToIntermediateImages + "IntermediateRedChannel/%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))
		lgn.debug(self.PathToIntermediateImages + "IntermediateGreenChannel/%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))
		lgn.debug(self.PathToIntermediateImages + "IntermediateBlueChannel/%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))

		RedChannelIntermediateImage.save(self.PathToIntermediateImages + "IntermediateRedChannel/%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))
		GreenChannelIntermediateImage.save(self.PathToIntermediateImages + "IntermediateGreenChannel/%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))
		BlueChannelIntermediateImage.save(self.PathToIntermediateImages + "IntermediateBlueChannel/%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))

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

		self.PathToIntermediateImages = PathToIntermediateImages

		lgn.info("Stacking batch %s" % (self.IndexOfWorkingBatch))

		lgn.debug("Loading images to process to memory")

		#Prepare list of images to stack
		for i in range(self.BatchSize):
			IndexOfImage: int = i + self.IndexOfWorkingBatch * self.BatchSize

			self.ListOfWorkingImages.append(
				self.ListOfImagePathsFiles[IndexOfImage]
			)
		
		# Backup GetDimensions in case dimensions have changed
		self.GetDimensions(
			self.ListOfWorkingImages[0]
		)
		
		DidLoadedImages: bool = self.LoadImages()
		if not DidLoadedImages:
			lgn.error("Did not load imaged correctly.")
			return False
		
		PercentageTracker: int = 0
		LastTime = time.time()
		LastDelta: float = 0
		
		IntermediateImageArrayChannel0 = numpy.zeros((self.WorkingImageSize[0], self.WorkingImageSize[1]), dtype=numpy.uint64)
		IntermediateImageArrayChannel1 = numpy.zeros((self.WorkingImageSize[0], self.WorkingImageSize[1]), dtype=numpy.uint64)
		IntermediateImageArrayChannel2 = numpy.zeros((self.WorkingImageSize[0], self.WorkingImageSize[1]), dtype=numpy.uint64)

		lgn.debug("Entering Main Stacking Loop <3")
		for x in range(self.WorkingImageSize[0]):
			for y in range(self.WorkingImageSize[1]):
				PixelIndex: int = self.WorkingImageSize[0]*y+x
				for ImageIndex in range(self.BatchSize):
					#Gamma corrected average of images m.sqrt((Imaged1 ** 2 + Image2 ** 2)/2)
					IntermediateImageArrayChannel0[tuple([y, x])] += (self.LoadedImages[ImageIndex][PixelIndex][0] ** 2)/self.BatchSize
					IntermediateImageArrayChannel1[tuple([y, x])] += (self.LoadedImages[ImageIndex][PixelIndex][1] ** 2)/self.BatchSize
					IntermediateImageArrayChannel2[tuple([y, x])] += (self.LoadedImages[ImageIndex][PixelIndex][2] ** 2)/self.BatchSize

			#Make time estimation cleaner
			if 100*x/self.WorkingImageSize[0] >= PercentageTracker:
				NowTime = time.time()
				NowDeltaTime: float = NowTime - LastTime
				CleansedDelta: float = (NowDeltaTime + LastDelta)/2

				ThisBatchesEstimatedTime: float = CleansedDelta * (100-PercentageTracker)
				FormattedEstimatedTime: str = FormatSecondsToString(ThisBatchesEstimatedTime)
				
				if PercentageTracker != 0:
					if self.DoStackingMultipleBatches:
						EstimatedTimeForRestOfBatches: float = 100*CleansedDelta*(self.EndingIndexForBatches-self.IndexOfWorkingBatch)
						OverallEstimatedTime: float = ThisBatchesEstimatedTime + EstimatedTimeForRestOfBatches
						FormattedEstimatedAllBatchesTime: str = FormatSecondsToString(OverallEstimatedTime)
						lgn.info("%s%% done for this batch, estimated time: %s" % (PercentageTracker, FormattedEstimatedAllBatchesTime))
					else:
						lgn.info("%s%% done, estimated time for this batch: %s" % (PercentageTracker, FormattedEstimatedTime))


				LastTime = NowTime
				LastDelta = NowDeltaTime
				PercentageTracker += 1
		
		#Correct for gamma
		lgn.debug("Correcting for gamma")
		GammaCorrectedArrayChannel0 = numpy.sqrt(IntermediateImageArrayChannel0)
		GammaCorrectedArrayChannel1 = numpy.sqrt(IntermediateImageArrayChannel1)
		GammaCorrectedArrayChannel2 = numpy.sqrt(IntermediateImageArrayChannel2)

		DebugImage = Image.fromarray(GammaCorrectedArrayChannel0, "F")

		#Normalising image, because dng images values goes from 0 to 2**10, whilst tiff requires from 0 to 1
		for x in range(self.WorkingImageSize[0]):
			for y in range(self.WorkingImageSize[1]):
				GammaCorrectedArrayChannel0[tuple([x, y])] = GammaCorrectedArrayChannel0[tuple([x, y])]/NormalisationFactor
				GammaCorrectedArrayChannel1[tuple([x, y])] = GammaCorrectedArrayChannel1[tuple([x, y])]/NormalisationFactor
				GammaCorrectedArrayChannel2[tuple([x, y])] = GammaCorrectedArrayChannel2[tuple([x, y])]/NormalisationFactor

		lgn.debug("Stacked, now saving intermediate image")
		SavedImage: bool = self.SaveIntermediateImage(GammaCorrectedArrayChannel0, GammaCorrectedArrayChannel1, GammaCorrectedArrayChannel2)
		if not SavedImage:
			return False
		
		lgn.debug("Intermediate image saved.")

		return True
	
	def StackAmountOfBatches(
			self: Self, 
			PathToSourceImages: str,
			PathToIntermediateImages: str,
			EndingIndex: int, 
			StartingIndex: int = 0
		) -> bool:

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
			DidStackingOnBatch: bool = self.StackBatchOfImages(self.PathToWorkingImages)
			if not DidStackingOnBatch:
				raise Exception

		lgn.info("Finished stacking batches of images.")

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
	LoadImages = 4

FunctionToCall: "FunctionsToCall" = FunctionsToCall.StackActualImages

def __main__(vArgs: list[str] = []) -> bool:
	global FunctionToCall

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

def StackTestImages(vArgs: list[str] = []) -> bool:
	global NormalisationFactor
	NormalisationFactor = 256

	SSLeXHandler: "SSLeX" = SSLeX("TestImages")

	SSLeXHandler.AddImagesToWorkingList("D:/Users/hatel/Pictures/SSLeXTest/")

	DidStack: bool = SSLeXHandler.StackBatchOfImages("D:/Users/hatel/Pictures/SSLeXIntermediate/")

	if not DidStack:
		return False
	
	return True
