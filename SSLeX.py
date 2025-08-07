"""SSLeX.py -> Stacked Super Long Exposured Images
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
Sizes: list[int] = [256, 256]
ChannelsPerPixel: int = 3

#Initiating logging
LOGLEVEL = lgn.DEBUG
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
	def __init__(
			self: Self
		):

		self.ListOfImagePaths: list[str] = []
		self.ListOfWorkingImages: list[str] = []
		self.PathToWorkingImages: str = ""

		self.BatchSize: int = 8
		self.IndexOfWorkingBatch: int = 0

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

		for ImageToLoad in self.ListOfWorkingImages:
			with Image.open(self.PathToWorkingImages + ImageToLoad) as LoadedImage:
				self.LoadedImages.append(list(LoadedImage.getdata()))
		
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
			IntermediateImage: str,
			PathToIntermediate: str,
		) -> bool:
		
		lgn.debug("Processing intermediate image for saving in my own proprietary format.")

		lgn.debug("Formatting done, now saving to disk")

		FileHandler = open(PathToIntermediate + "Intermediate.intermediate", "w+")

		FileHandler.write(IntermediateImage)

		FileHandler.close()

		lgn.debug("Saved to disk")

		return True

	def LoadIntermediateImage(
			self: Self,
			PathToIntermediateImage: str,
			NameOfImage: str,
			Sizes: list[int],
			ChannelsPerPixel: int,
		):
		
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
					TemporaryPixel.append(int(Values[ValueTracker]))
					ValueTracker += 1
				TemporaryImage.putpixel([x,y], tuple(TemporaryPixel))

		TemporaryImage.show()
		TemporaryImage.save(PathToIntermediateImage + "Image.png")

	def StackBatchOfImages(
			self: Self,
			PathToIntermediateImages: str
		) -> bool:

		lgn.debug("Loading images to process to memory")
		#Prepare list of images to stack
		for i in range(self.BatchSize):
			IndexOfImage: int = i + self.IndexOfWorkingBatch * self.BatchSize
			self.ListOfWorkingImages.append(
				self.ListOfImagePathsFiles[IndexOfImage]
			)
		
		#Backup GetDimensions in case dimensions have changed
		# self.GetDimensions(
		# 	self.ListOfWorkingImages[0]
		# )
		
		DidLoadedImages: bool = self.LoadImages()
		if not DidLoadedImages:
			lgn.error("Did not load imaged correctly.")
			return False
		
		PercentageTracker: int = 0
		MaxPixelIndex: int = self.WorkingImageSize[0]*self.WorkingImageSize[1]
		LastTime = time.time()
		LastDelta: float = 0
		IntermediateString: str = ""

		lgn.debug("Entering Main Stacking Loop <3")
		for PixelIndex in range(MaxPixelIndex):
				
			TemporaryPixel: list[float] = [0 for _ in range(self.ChannelsPerPixel)]

			#Process stacking
			for ImageIndex in range(len(self.ListOfWorkingImages)):
				for ChannelIndex in range(self.ChannelsPerPixel):
					TemporaryPixel[ChannelIndex] += self.LoadedImages[ImageIndex][PixelIndex][ChannelIndex]/self.BatchSize
			
			for ChannelIndex in range(self.ChannelsPerPixel):
				IntermediateString += str(int((1024**2)*TemporaryPixel[ChannelIndex])) + ":"

			#Make time estimation cleaner
			if 100*PixelIndex/MaxPixelIndex >= PercentageTracker:
				NowTime = time.time()
				NowDeltaTime: float = NowTime - LastTime
				CleansedDelta: float = (NowDeltaTime + LastDelta)/2

				EstimatedTime: str = FormatSecondsToString(CleansedDelta * (100-PercentageTracker))
				
				if PercentageTracker != 0:
					lgn.debug("%s%% done, estimated time: %s" % (PercentageTracker, EstimatedTime))

				LastTime = NowTime
				LastDelta = NowDeltaTime
				PercentageTracker += 1
		
		lgn.debug("Stacked, now saving intermediate image")
		SavedImage: bool = self.SaveIntermediateImage(IntermediateString, PathToIntermediateImages)
		if not SavedImage:
			return False
		
		lgn.debug("Intermediate image saved.")

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

	TemporaryImage = numpy.zeros([Sizes[0], Sizes[1], ChannelsPerPixel], dtype=numpy.uint64)
	for x in range(Sizes[0]):
		for y in range(Sizes[1]):
			for ChannelIndex in range(ChannelsPerPixes):
				RandomValue: int = RandomHandler.RandomInt(None, 0, MaxSubPixelValue)

				TemporaryImage[x][y][ChannelIndex] = math.floor(256*(Sizes[1]*x + y*IndexOfImage/8)/(Sizes[0]*Sizes[1]))

	NumpyArray = numpy.array(TemporaryImage)
	OutputImage = Image.fromarray(NumpyArray, "RGB")

	OutputImage.save(OutputPath + OutputName + ".png")

class FunctionsToCall(Enum):
	DoNothing = 0
	CreateRandomisedImages = 1
	StackImages = 2
	LoadImages = 3

FunctionToCall: "FunctionsToCall" = FunctionsToCall.StackImages

def __main__(vArgs: list[str] = []) -> bool:
	global FunctionToCall

	ReturnValue: bool
	if FunctionToCall == FunctionsToCall.DoNothing:
		ReturnValue= DoNothing(vArgs)
	elif FunctionToCall == FunctionsToCall.CreateRandomisedImages:
		ReturnValue = CreateRandomisedImages(vArgs)
	elif FunctionToCall == FunctionsToCall.StackImages:
		ReturnValue = StackImages(vArgs)
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
		CreateRandomisedImage("D:/Users/hatel\Pictures/SSLeXTest/", str(i), i)
		lgn.debug("Created %sth image" % (i))
	
	return True

#Normal stack
def StackImages(vArgs: list[str] = []) -> bool:
	SSLeXHandler: "SSLeX" = SSLeX()

	SSLeXHandler.AddImagesToWorkingList("D:/Users/hatel/Pictures/SSLeX1024/")

	DidStack: bool = SSLeXHandler.StackBatchOfImages("D:/Users/hatel/Pictures/SSLeXIntermediate/")

	if not DidStack:
		return False
	
	return True
