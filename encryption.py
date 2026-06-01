import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# LFSR set 
seed_bits = [1, 0, 1, 1, 0, 0, 1, 1] 
taps = [0, 3] 
def lfsr_key_stream(shape): 
sr, bits = seed_bits[:], [] 
for _ in range(np.prod(shape) * 8): 
xor = sum([sr[t] for t in taps]) % 2 
bits.append(sr[-1]) 
sr = [xor] + sr[:-1] 
return np.packbits(bits)[:np.prod(shape)].reshape(shape).astype(np.uint8) 

#load image 
img = cv2.imread('lena.png') 
if img is None: raise ValueError("loading faild") 
original = img.copy() 
rows, cols, _ = img.shape 
# Confusion using  LFSR 
key = lfsr_key_stream((rows, cols, 3)) 
confused_img = np.bitwise_xor(img, key) 

# Diffusion 
diffused_img = confused_img.copy() 
for r in range(1, rows): 
for c in range(1, cols): 
diffused_img[r, c] ^= diffused_img[r-1, c] ^ diffused_img[r, c-1] 

#reverse diffusion and confusion 
restored_img = diffused_img.copy() 
for r in range(rows-1, 0, -1): 
for c in range(cols-1, 0, -1): 
restored_img[r, c] ^= restored_img[r-1, c] ^ restored_img[r, c-1] 
decrypted_img = np.bitwise_xor(restored_img, key) 
titles = ["Original", "Confused", "Diffused", "Decrypted"] 
images = [original, confused_img, diffused_img, decrypted_img] 

#diplay 
plt.figure(figsize=(12, 4)) 
for i, (title, img) in enumerate(zip(titles, images)): 
plt.subplot(1, 4, i+1) 
plt.title(title) 
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
plt.axis('off') 
plt.tight_layout() 
plt.show()
