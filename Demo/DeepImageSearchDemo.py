# Importing the proper classes
from DeepImageSearch import Index,LoadData,SearchImage

# load the Images from the Folder (You can also import data from multiple folders in python list type)
image_list = LoadData().from_folder(['images','wiki-images'])

# For Faster Serching we need to index Data first, After Indexing all the meta data stored on the local path
Index(image_list).Start()

# for searching, you need to give the image path and the number of the similar image you want
SearchImage().get_similar_images(image_path=image_list[0],number_of_images=5)

# If you want to plot similar images you can use this method, It will plot 16 most similar images from the data index
SearchImage().plot_similar_images(image_path = image_list[0])

