import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("タブUIでサイト埋め込み表示")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["gimy日本動漫", "baseball-ptt", "mlb-ptt", "nba-ptt", "shadowverse-ptt"])

with tab1:
    st.subheader("gimy日本動漫")
    components.iframe("https://gimy.tv/genre/4-%E6%97%A5%E6%9C%AC--/by/time.html", height=1000, width=None, scrolling=True)

with tab2:
    st.subheader("baseball-ptt")
    components.iframe("https://www.ptt.cc/bbs/Baseball/index.html", height=1000, width=None, scrolling=True)

with tab3:
    st.subheader("mlb-ptt")
    components.iframe("https://www.ptt.cc/bbs/MLB/index.html", height=1000, width=None, scrolling=True)

with tab4:
    st.subheader("nba-ptt")
    components.iframe("https://www.ptt.cc/bbs/NBA/index.html", height=1000, width=None, scrolling=True)

with tab5:
    st.subheader("shadowverse-ptt")
    components.iframe("https://www.ptt.cc/bbs/Shadowverse/index.html", height=1000, width=None, scrolling=True)