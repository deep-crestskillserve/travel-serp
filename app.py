import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from agents.agent import run_agent

def initialize_session():
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            color: #333;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        ''', unsafe_allow_html=True)

def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">AI Travel Agent 🏨</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your travel query and get flight and hotel information:</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        'Travel Query',
        height=200,
        key='query',
        placeholder='Type your travel query here...',
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return user_input

def process_query(user_input):
    if user_input:
        try:
            thread_id = st.session_state.thread_id
            query = f"THREAD_ID::{thread_id}::{user_input}"
            result = run_agent(query)  # Synchronous call
            st.subheader('Travel Information')
            st.write(result)
            st.session_state.travel_info = result
        except Exception as e:
            # stbur
            st.error(f'Error: {e}')
    else:
        st.error('Please enter a travel query.')

def main():
    initialize_session()
    render_custom_css()
    user_input = render_ui()

    if st.button('Get Travel Information'):
        process_query(user_input)

if __name__ == '__main__':
    main()