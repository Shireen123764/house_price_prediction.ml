import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
import plotly.express as px

st.set_page_config(layout="wide")

select = option_menu(
    menu_title=None,
    options=["Home","Predict Price"],
    orientation="horizontal"
)


df=pd.read_csv("house_price_prediction.csv")

if select == "Home":
    st.dataframe(df)

    col5,col6,col7,col8 = st.columns(4)
    with col5:
        st.metric("Total location ",df["location"].nunique())
    with col6:
        st.metric("Average Prices",round(df["price"].mean(),2))
    with col7:
        st.metric("Maximum Price ",round(df["price"].max(),2))
    with col8:
        st.metric("Minimum Price ",round(df["price"].min(),2))

    st.title("location wise (model) price trend!")

    loc = df.groupby("location")["price"].mean().reset_index()

    fig=px.line(
        loc,
        x="location",
        y="price",
        title="Price Trend"
    )

    st.plotly_chart(fig)


if select== "Predict Price":

    st.title("Predict the price house")
   

    le=LabelEncoder()

    df['location']=le.fit_transform(df['location'])

    x=df[["location","area_sqft","rooms"]]
    y=df.price

    model=linear_model.LinearRegression()
    #st.write(df.isnull().sum())
    #st.write(df[df["rooms"].isnull()])
    model.fit(x,y)

    col9,col10=st.columns(2)
    with col9:
        st.info("by Selecting location, area_sqft, and rooms")
        rooms=st.selectbox("Enter room:",[1,2,3,4,5,6,7,8,9,10])
        area_sqft=int(st.number_input("Enter area_sqft:",min_value=1000))
        location=st.selectbox("select location ",le.classes_)
        pred=st.button("Predict")


    with col10:
        ro=le.transform([location])[0]
        predicted_price=model.predict([[ro,area_sqft,ro]])
        score=model.score(x,y)
        acuracy=int(score*100)
        st.metric("Accuracy of Current prices!",acuracy,"%")
        if pred:
           
            if rooms:
                ro=le.transform([location])[0]
                predicted_price=model.predict([[ro,area_sqft,ro]])
                st.balloons()
                score=model.score(x,y)
                acuracy=int(score*100)
                st.subheader("How accurate the predicted price is!")
                st.info(f"{acuracy}%")
                st.subheader("Here is your Predicted Price for house")
                predic=int(predicted_price[0])
                st.info(predic)

            existing_data=pd.read_csv("house_price_prediction.csv")
            new_data=pd.DataFrame({"location":[location],"area_sqft":[area_sqft],"room":[rooms],"price":[predic]})
            updated_data=pd.concat([existing_data,new_data])
            updated_data.to_csv("house_price_prediction.csv",index=False)