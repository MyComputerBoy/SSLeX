"""SSLeX.py -> Stacked Super Long Exposured Images
"""

import math
import os
import sys
from PIL import Image
import Scipy.misc
import numpy
from typing import Self
import struct

sys.path.insert(0, "C:/Users/Kim Chemnitz/Documents/GitHub/Hash/")
import MCGRandom as Random

RandomHandler = Random.MCGRandom()

def ByteToChars(Input: int) -> str:
	LowerChar = RandomHandler.ConvertableCharList[Input % 64]
	HigherChar = RandomHandler.ConvertableCharList[math.floor(Input/64)]

	return str(HigherChar + LowerChar)

class SSLeX():
	def __init__(
			self: Self
		):

		self.ListOfImagePaths: list[str] = []
		self.ListOfWorkingImages: list[str] = []

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

		return True
	
	def LoadImages(
			self: Self,
		) -> bool:

		for ImageToLoad in self.ListOfWorkingImages:
			with Image.open(ImageToLoad) as LoadedImage:
				self.LoadedImages.append(list(LoadedImage.getdata()))
		
		return True

	def GetDimensions(
			self: Self,
			ImagePath: str
		) -> bool:

		with Image.open(ImagePath) as WorkingImage:
			self.WorkingImageSize = WorkingImage.size
		
		return True

	def SaveIntermediateImage(
			self: Self,
			IntermediateImage: list[list[float]],
			PathToIntermediate: str,
		) -> bool:
		
		OutputString: str = ""

		for PixelIndex in range(len(IntermediateImage)):
			for ChannelIndex in range(self.ChannelsPerPixel):
				WorkingValue: float = IntermediateImage[PixelIndex][ChannelIndex]
				ByteArrayRepresentation: bytearray = bytearray(struck.pack("d", WorkingValue))
				for WorkingByte in ByteArrayRepresentation:
					OutputString += ByteToChars(WorkingByte)

		FileHandler = open(PathToIntermediate + "Intermediate.intermediate", "w+")

		FileHandler.write(OutputString)

		FileHandler.close()

		return True

	def StackBatchOfImages(
			self: Self,
			PathToIntermediateImages: str
		) -> bool:

		for i in range(self.BatchSize):
			IndexOfImage: int = i + self.IndexOfWorkingBatch * self.BatchSize
			self.ListOfWorkingImages.append(
				self.ListOfImagePathsFiles[IndexOfImage]
			)
		
		#Backup GetDimensions in case dimensions have changed
		self.GetDimensions(
			self.ListOfWorkingImages[0]
		)
		
		DidLoadedImages: bool = self.LoadImages()
		if not DidLoadedImages:
			return False
		
		IntermediateStackedImage: list[list[float]] = []

		for PixelIndex in range(
			self.WorkingImageSize[0]*self.WorkingImageSize[1]
			):
				
			TemporaryPixel: list[float] = [0 for _ in range(self.ChannelsPerPixel)]

			for ImageIndex in range(len(self.ListOfWorkingImages)):
				for ChannelIndex in range(self.ChannelsPerPixel):
					TemporaryPixel[ChannelIndex] += self.LoadedImages[ImageIndex][PixelIndex][ChannelIndex]/self.BatchSize
			
			IntermediateStackedImage.append(TemporaryPixel)
		
		SavedImage: bool = self.SaveIntermediateImage(IntermediateStackedImage, PathToIntermediateImages)
		if not SavedImage:
			return False

		return True

