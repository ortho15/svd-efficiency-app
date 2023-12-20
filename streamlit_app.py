import streamlit as st
import func
from PIL import Image
import os


# Webpage Config
st.set_page_config(page_title='Image SVD Effiency Demonstration')

# Body
st.markdown("# Image SVD Effiency Demonstration")

user_img = st.file_uploader(label="Upload your own jpg image to test!", type=["png", "jpg"])
image = None
if user_img:
    with open("./assets/user.jpg", 'wb') as f:
        f.write(user_img.getvalue())
    image = "./assets/user.jpg"
else:
    image = "./assets/plane.jpg"

st.markdown("## Original Image")
st.image(image)

st.markdown("## After SVD")

R,G,B = func.jpg_to_RGB(image)
three = {
    'R': R,
    'G': G,
    'B': B
}
og_size = R.shape

# CORE

l, r = st.columns([0.8, 0.2])

osize = round(os.stat(image).st_size / 1024)
rsize = round(os.stat("./assets/reduced.png").st_size / 1024)
r.write(f"""
Original dimensions: {og_size}\n
Original file size: {osize} KiB \n
Reduced file size: {rsize} KiB \n
Percentage difference: {round(((rsize - osize) / osize)*100)} %
""")

cache_SVt = func.eigens(R,G,B) # number of singular values ordered from biggest to smallest
top = max(og_size)
# rank = min(len(rgb_eigens['R'][0]), len(rgb_eigens['G'][0]), len(rgb_eigens['B'][0]))
chosen = st.slider("rank", 1, og_size[1], og_size[1]-20, 10)
func.compress(chosen, three, cache_SVt)
l.image('./assets/reduced.png')