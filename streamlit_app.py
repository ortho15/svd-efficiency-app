import streamlit as st
import func

# Webpage Config
st.set_page_config(page_title='Image SVD Effiency Demonstration')
#, page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

# Globals
image = "./assets/main.jpg"

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

rbg_eigens = func.eigens(R,G,B) # number of singular values ordered from biggest to smallest
rank = min(len(rbg_eigens['R'][0]), len(rbg_eigens['G'][0]), len(rbg_eigens['B'][0]))
start = rank-20
chosen = st.slider("rank", 1, rank, start, 10)
R = func.compress(chosen, three)
l.image(R)

# TESTING!!!
# TESTING!!!
