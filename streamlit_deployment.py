from PIL import Image
import streamlit as st

# set title
st.title("My First Streamlit App")

# add sub header
st.subheader("this is the sub header")

# adding image
image = Image.open("1267203.jpg")
st.image(image, use_column_width=True)

# adding writing and markdown
st.write("writing a text !")
st.markdown("markdown cell ")

# adding a success
st.success("congrats this cell ran !")

# adding info , warning , error
st.info(" This is the information cell")
st.warning(" warning cell !")
st.error("error cell !")

# help
st.help(range)

# pandas data frame
import numpy as np 
import pandas as pd 

st.subheader('this is a dataframe of random numbers between 10,20')
dataframe = np.random.rand(10,20)
st.dataframe(dataframe)

st.text('another dataframe creation')
df = pd.DataFrame(np.random.rand(10,20),columns = ('col %d' %i for i in range(20)))
st.dataframe(df.style.highlight_max(axis = 1))

# display charts and graphs 

data = pd.DataFrame(np.random.randn(20,3), columns = ['a','b','c'])

st.text('line chart')
st.line_chart(data)

st.text('area chart')
st.area_chart(data)

st.text('bar chart')
st.bar_chart(data)

# using matplot
from pyforest import *
st.set_option('deprecation.showPyplotGlobalUse', False)

st.text('pyplot module')
arr = np.random.normal(1,1,size = 100)
plt.hist(arr,bins = 20)
st.pyplot()

# using plotly
import plotly
import plotly.figure_factory as ff 

x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) - 5

st.text("plotly graph")
hist_data = [x1,x2,x3]
labels = ['group 1','group 2','group 3']
fig = ff.create_distplot(hist_data,labels,bin_size = [0.5,0.25,0.6])
st.plotly_chart(fig,use_container_width = True)

# adding buttons 
st.text('button types')
if st.button('hey'):
    st.write('hello')

detail = st.radio('are you single ?',('yes','no'))
if detail == 'yes':
    st.write('eligible')
else:
    st.write('Not eigible')

ans = st.selectbox('are you married ? ',('yes','no'))
st.write('are you married ? : ',ans )

multans = st.multiselect('select year ? multi select ',(20,30,50,70))
st.write('years are : ',multans)

# adding slider
st.text('slider !')
age = st.slider('how old are you ?',0,90,10)
st.write('I am ',age )

value = st.slider('select range',0,90,(10,20))
st.write('range : ',value )

# number input
num = st.number_input('input number ')
st.write('Number : ',num)

# adding file or uploading
file = st.file_uploader("Choose a file",type = 'csv')
if file is not None:
    data = pd.read_csv(file,header = 0)
    st.write(data)
    st.success("file uploaded !")
else :
    st.markdown("upload csv file")


# color picker
color = st.color_picker('pick your color','#00f900')
st.write("color is : ",color) 

# adding sidebar
st.sidebar.selectbox('choose model :' ,('model 1','model 2','model 3'))

# progress bar for tracking
st.text('progress bar 1')
import time
my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete+1)

st.text('alternative progress')
with st.spinner("please wait .........") :
    time.sleep(5)
st.balloons()
st.success("complete !")
