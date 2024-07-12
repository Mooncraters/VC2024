# preprocess

## load the image: K

```python
def load_the_image(path)
	'''
	return two list, all_image and all_label
	'''
```



## crop: K

```python
def crop(all_image)
	'''
	return all_image after crop.
	crop by the x1, x2, y1, y2. and they are read from Train.csv
	'''
```



## data enhancement: C

```python
def data_enhancement(all_image)
	'''
	return a set of image
	'''
```



## attention

no need for blur, equalize or something of preprocess of image, do it in the specific function of classifier and training  

# classifier

a part need to explore

## PCA to get feature: K

try use PCA to get feature. hint: add 'visualize' parameter

there some problem to fit

- can we use these features to directly recognize? without use machine learning

## other ways to get feature: X, D, K

to be explored

## color detect: D

```python
def color_detect(image)
	'''
	paramenter is a image read by cv2.imread()
	return the class of Color
	'''
```





## shape detect: K

```python
def color_detect(image)
	'''
	paramenter is a image read by cv2.imread()
	return the class of Shape
	'''
```



## some problems

- only detect by edge can easily make mistakes
- maybe can use the features from PCA or other methods to detect
- how to set confidence value
- set distance of features, to correct error?
- use an overall feature.(I don't no what specific thing it refers to ) 

## content detect

do it after we have fix the problem below.

# recognition

## train by different class, and store the model: C

## judge the class of image and select which model to use: C

## fit based on the result of fit: C

## test

## deep learning

may be