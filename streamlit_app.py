import streamlit as st
import func

# Webpage Config
st.set_page_config(page_title='Image SVD Effiency Demonstration')
#, page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

# Globals
image = "./assets/plane.jpg"

# Body

st.markdown("# Image SVD Effiency Demonstration")

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

r.write(f"""matrix_to_img
Original size: {og_size}
""")

rgb_eigens = func.eigens(R,G,B) # number of singular values ordered from biggest to smallest
st.write(rgb_eigens['B'][0])
top = max(og_size)
# rank = min(len(rgb_eigens['R'][0]), len(rgb_eigens['G'][0]), len(rgb_eigens['B'][0]))
chosen = st.slider("rank", 1, og_size[1], og_size[1]-20, 10)
func.compress(chosen, three)
l.image('./assets/reduced.png')

# TESTING!!!
# TESTING!!!
