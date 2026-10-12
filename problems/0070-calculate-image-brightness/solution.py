import numpy as np 

def calculate_brightness(img):
	# Handling edge cases
	if not img or len(img) == 0:
		return -1
	# Check for inconsistent row lengths 
	row_lens = [len(img[i]) for i in range(0, len(img))]
	if len(set(row_lens)) > 1:
		return -1
	
	img_np = np.array(img)
	if np.max(img_np) > 255 or np.min(img_np) < 0:
		return -1 

	return np.mean(img_np)
