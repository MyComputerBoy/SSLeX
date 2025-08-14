# SSLeX
Stacked Super Long Exposured Images
A project for stacking multiple images for super high quality images from multiple images.
(Still early in development)

How to use SSLeX

SSLeX (Stacked Super Long eXposures) uses a class to store and work on everything, and to initiate a class you need to supply a name for the project as to differentiate on different projects as nescessary.

Example:

SSLeXHandler = SSLeX("Hello World!")


Then there are three main functions to use:
AddImagesToWorkingList(PathsToImages: str) -> bool, Adds path to images to stack, the folder should be empty except the images you want to stack (Kind of redundant if you wanna stack more batches than one
StackAmountOfBatches(self: Self, PathToSourceImages: str,PathToIntermediateImages: str,EndingIndex: int, StartingIndex: int = 0) -> bool, Stacks StartingIndex to EndingIndex batches (a batch is stacking 8 pictures to one)
StackIntermediateImages(self: Self, PathToSourceImages: str,c: str,EndingIndex: int,StartingIndex: int = 0) -> bool, Stacks Intermediate images from StartingIndex to EndingIndex of batches, just supply basefolder that contains all channel subfolders as source path

Example:

DidStacking: bool = SSLeXHandler.StackAmountOfBatches(PathToSourceImages="C:/Users/MyUsername/Pictures/SourceImages/", PathToIntermediateImages="C:/Users/MyUsername/Pictures/IntermediateImages/", EndingIndex=16, StartingIndex=0)
if not DidStacking:
  raise Exceptiong("Could not stack images properly.")

And for stacking intermediate images it's mostly the same

Example:

DidIntermediateStacking: bool = SSLeXHandler.StackIntermediateImages(PathToSourceImages="C:/Users/MyUsername/Pictures/Intermediate/", PathToSourceImages="C:/Users/MuUsername/Pictures/NextIntermediate/", EndingIndex=2, StartingIndex=0)
if not DidIntermediateStacking:
  raise Exception("Could not stack intermediate images properly)
