"""SSLeX.py -> Stacked Super Long Exposured Images
Refactoring branch
Main user classes:

SSLeX(NameOfImages: str) -> Main class of Stacked Super Long eXposured images, NameOfImages is to differentiate between different SSLeX intermediate images
	Main user functions:

	AddImagesToWorkingList(PathsToImages) -> bool, Adds path to images to stack, the folder should be empty except the images you want to stack
	StackAmountOfBatches(
			self: Self, 
			PathToSourceImages: str,
			PathToIntermediateImages: str,
			EndingIndex: int, 
			StartingIndex: int = 0
	) -> bool, Stacks StartingIndex to EndingIndex batches
	StackIntermediateImages(
		self: Self,
		PathToSourceImages: str,
		PathToIntermediateImages: str,
		EndingIndex: int,
		StartingIndex: int = 0
	) -> bool, Stacks Intermediate images from StartingIndex to EndingIndex of batches, just supply basefolder that contains all channel subfolders as source path
"""

import math
import os
from PIL import Image
import PIL
import numpy
from typing import Self
import time
import logging as lgn			#Logging for custom exceptions
import gc

#Initiating logging
LOGLEVEL = lgn.DEBUG
lgn.basicConfig(format="%(levelname)s: %(message)s", level=lgn.DEBUG)
lgn.getLogger().setLevel(LOGLEVEL)

def FormatSecondsToString(InputSeconds: float) -> str:
	Seconds = int(InputSeconds % 60)
	Minutes = int(math.floor(InputSeconds/60) % 60)
	Hours   = int(math.floor(InputSeconds/3600))

	return "%sH %sM %sS" % (Hours, PadStringOnLeft(Minutes, 2), PadStringOnLeft(Seconds, 2))

def PadNumber(InputNumber: int, TargetLength: int) -> str:
	OutputString: str = str(InputNumber)
	while len(OutputString) < TargetLength:
		OutputString = "0" + OutputString
	
	return OutputString

def PadStringOnLeft(InputString: str, TargetLength: int) -> str:
	OutputString: str = str(InputString)
	while len(OutputString) < TargetLength:
		OutputString = " " + OutputString
	
	return OutputString

class SSLeX():
	"""SSLeX(NameOfImages: str) -> Main class of Stacked Super Long eXposured images, NameOfImages is to differentiate between different SSLeX intermediate images
	Main user functions:

	AddImagesToWorkingList(PathsToImages) -> bool, Adds path to images to stack, the folder should be empty except the images you want to stack
	StackAmountOfBatches(
			self: Self, 
			PathToSourceImages: str,
			PathToIntermediateImages: str,
			EndingIndex: int, 
			StartingIndex: int = 0
	) -> bool, Stacks StartingIndex to EndingIndex batches
	StackIntermediateImages(
		self: Self,
		PathToSourceImages: str,
		PathToIntermediateImages: str,
		EndingIndex: int,
		StartingIndex: int = 0
	) -> bool, Stacks Intermediate images from StartingIndex to EndingIndex of batches, just supply basefolder that contains all channel subfolders as source path
	"""

	def __init__(
			self: Self,
			NameOfImages: str,
		):

		self.Name = NameOfImages

		#Definitions of different paths needed
		self.ListOfImagePaths: list[str] = []
		self.ListOfWorkingImages: list[str] = []
		self._BaselineWorkingImagesPath_: str = ""
		self.PathToWorkingImages: str = ""
		self.PathToIntermediateImages: str = ""

		#Definitions of some information about the working process
		self.BatchSize: int = 8
		self.IntermediateIndex: int = 0
		self.IndexOfWorkingBatch: int = 0

		self.DoStackingMultipleBatches: bool = False
		self.DoStackingIntermediateImages: bool = False

		self.StartingIndexForBatches: int = 0
		self.EndingIndexForBatches: int = 0

		self.WorkingImageSize: list[int] = [8160,6120]	#For 50MP RAW images
		self.BaselineChannelsPerPixel: int = 3
		self.WorkingChannelsPerPixel: int = 3
		self.IntermediateScalar: float = 2**32
		self.NormalisationFactor: float = 2**10

		self.LoadedImages: list[list[list[int]]] = []

		#Definitions of percentage tracking variables, for printing how long the process is along the stacking
		self._PercentageTracker_: int = 0
		self._LastTime_: float = 0
		self._LastDelta_: float = 0

		self.VerboseDebugging: bool = False
	
	def AddImagesToWorkingList(
			self: Self, 
			PathsToImages: str
		) -> bool:

		self.ListOfImagePathsFiles = os.listdir(PathsToImages)

		if len(self.ListOfImagePathsFiles) < 2:
			raise Exception("The specified folder needs to contain at least 2 images to stack.")
		
		lgn.debug("Added %s images to self.ListOfImagePathFiles." % (len(self.ListOfImagePathsFiles)))

		self.PathToWorkingImages = PathsToImages

		return True
	
	def LoadImages(
			self: Self,
		) -> bool:

		CheckingImage = self.ListOfWorkingImages[0].split(".")[1].lower()

		if CheckingImage == "tif" or CheckingImage == "png" or CheckingImage == "tiff":
			for ImageToLoad in self.ListOfWorkingImages:
				lgn.debug("Loading image %s" % (ImageToLoad))
				with Image.open(self.PathToWorkingImages + ImageToLoad) as LoadedImage:
					self.LoadedImages.append(list(LoadedImage.getdata()))
		elif CheckingImage == "intermediate":
			for ImageToLoad in self.ListOfWorkingImages:
				lgn.debug("Loading image %s" % (ImageToLoad))
				DidLoadImage = self.LoadIntermediateImage(self.PathToWorkingImages, ImageToLoad, self.WorkingImageSize, self.WorkingChannelsPerPixel)
				if not DidLoadImage:
					raise Exception
		else:
			raise TypeError("Imagetype %s is not supported." % (CheckingImage))
		
		lgn.debug("Loaded images :3")
		return True

	def GetDimensions(
			self: Self,
			ImagePath: str
		) -> bool:

		try:
			with Image.open(self.PathToWorkingImages + ImagePath) as WorkingImage:
				self.WorkingImageSize = WorkingImage.size
		except PIL.UnidentifiedImageError:
			raise Exception("Unsupported image format (PIL does not recognise the format)")
		
		return True

	def SaveIntermediateImage(
			self: Self,
			IntermediateImage: numpy.ndarray,
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
				IntermediateChannelPath: str = self.PathToIntermediateImages + "Intermediate%sChannel/" % (ChannelIndex)

				#Make sure the sub folder for this Channel in there, so you don't have to make all of them manually <3
				if not os.path.exists(IntermediateChannelPath):
					os.makedirs(IntermediateChannelPath)
					lgn.debug("Created folder at \"%s\"." % (IntermediateChannelPath))
				
				IntermediateImages[ChannelIndex].save(IntermediateChannelPath + "%sBatch%s.tiff" % (self.Name, self.IndexOfWorkingBatch))

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
					TemporaryPixel.append(int(int(Values[ValueTracker])/self.IntermediateScalar))
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
		
		self.PathToIntermediateImages = PathToIntermediateImages

		lgn.info("Stacking batch %s" % (self.IndexOfWorkingBatch))

		lgn.debug("Loading images to process to memory")

		lgn.debug("ListOfImagePathsFiles: %s" % (self.ListOfImagePathsFiles))

		self.ListOfWorkingImages = []
		#Prepare list of images to stack
		try:
			for i in range(self.BatchSize):
				IndexOfImage: int = i + self.IndexOfWorkingBatch * self.BatchSize

				self.ListOfWorkingImages.append(
					self.ListOfImagePathsFiles[IndexOfImage]
				)
		except IndexError:
			raise IndexError("There are not enough images left for a full batch (%s of images, a full batch is %s)" % (len(self.ListOfImagePathsFiles), self.BatchSize))
		
		# Backup GetDimensions in case dimensions have changed
		self.GetDimensions(self.ListOfWorkingImages[0])
		
		DidLoadedImages: bool = self.LoadImages()
		if not DidLoadedImages:
			lgn.error("Did not load imaged correctly.")
			return False
		
		self.ResetPercentageTracker()

		MaxPixelIndex = self.WorkingImageSize[0] * self.WorkingImageSize[1]

		lgn.debug("Entering Main Stacking Loop <3")

		#The main stacking
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
							#Gamma corrected average of images m.sqrt((Image1 ** 2 + Image2 ** 2)/2)
							IntermediateImageArray[tuple([ChannelIndex, x, y])] += (self.LoadedImages[ImageIndex][PixelIndex][ChannelIndex] ** 2)/self.BatchSize
					self.PrintPercentage(PixelIndex, MaxPixelIndex)

		#Correct for gamma
		lgn.debug("Correcting for gamma")
		GammaCorrectedArray = numpy.sqrt(IntermediateImageArray)

		self.ResetPercentageTracker()

		#Normalising image, because dng images values goes from 0 to 2**10, whilst tiff requires from 0 to 1
		if not self.DoStackingIntermediateImages:
			lgn.debug("Normalising image")
			for x in range(self.WorkingImageSize[0]):
				for y in range(self.WorkingImageSize[1]):
					for ChannelIndex in range(self.WorkingChannelsPerPixel):
						GammaCorrectedArray[tuple([ChannelIndex, x, y])] = GammaCorrectedArray[tuple([ChannelIndex, x, y])]/self.NormalisationFactor
					self.PrintPercentage(self.WorkingImageSize[1]*x+y, self.WorkingImageSize[0] * self.WorkingImageSize[1])

		lgn.debug("Stacked, now saving intermediate image")
		SavedImage: bool = self.SaveIntermediateImage(GammaCorrectedArray)
		if not SavedImage:
			return False
		
		lgn.debug("Intermediate image saved.")

		lgn.debug("Deleting LoadedImages")
		self.LoadedImages = []
		lgn.debug("Images deleted.")

		lgn.debug("Stacking of this batch is done.")

		return True
	
	def ResetPercentageTracker(
			self: Self
	) -> None:
		self._LastTime_ = time.time()
		self._LastDelta_ = 0
		self._PercentageTracker_ = 0

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
					lgn.info("%s%% done for this batch, estimated time: %s" % (PadStringOnLeft(self._PercentageTracker_, 2) , FormattedEstimatedAllBatchesTime))
				else:
					lgn.info("%s%% done, estimated time for this batch: %s" % (PadStringOnLeft(self._PercentageTracker_, 2) , FormattedEstimatedTime))
	
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
