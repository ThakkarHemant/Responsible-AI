"""
Streamlit Frontend for Memory-Augmented Legal Chatbot
Shows real-time visualization of Short, Medium, and Long-term memory
"""

import streamlit as st
from datetime import datetime
import json

# Import the enhanced chatbot (ensure legal_chatbot.py is in the same directory)
try:
    from legal_chatbot import MemoryAugmentedLegalChatbot, Config
except ImportError:
    st.error(" Please ensure 'legal_chatbot.py' is in the same directory!")
    st.stop()

# ============================================================================ #
# PAGE CONFIGURATION
# ============================================================================ #

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .memory-box {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #000000;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .chat-user {
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-bot {
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================ #
# SESSION STATE INITIALIZATION
# ============================================================================ #

if 'chatbot' not in st.session_state:
    with st.spinner(" Initializing Legal AI Assistant..."):
        st.session_state.chatbot = MemoryAugmentedLegalChatbot()
        st.session_state.chat_history = []
        st.session_state.user_id = "user_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    #st.success(" Chatbot initialized!")

# ============================================================================ #
# SIDEBAR - MEMORY VISUALIZATION
# ============================================================================ #

with st.sidebar:
    st.markdown("###  Legal AI Assistant")
    st.markdown("---")
    
    # User info
    st.markdown(f"**Session ID:** `{st.session_state.user_id[:15]}...`")
    st.markdown(f"**Started:** {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    st.markdown("###  Memory System Overview")
    
    # Get memory states
    memory_states = st.session_state.chatbot.get_memory_states()
    
    # Short-term memory
    with st.expander(" Short-term Memory (Recent Turns)", expanded=True):
        short_count = memory_states['short_term']['count']
        st.metric("Active Turns", short_count, delta=f"Max: {Config.SHORT_TERM_MAX_TURNS}")
        
        if short_count > 0:
            st.markdown("**Recent conversations:**")
            for turn in memory_states['short_term']['turns'][-3:]:
                st.markdown(f"🔹 Turn {turn['turn']}: *{turn['user'][:40]}...*")
        else:
            st.info("No recent conversation yet")
    
    # Medium-term memory
    with st.expander(" Medium-term Memory (Summaries)", expanded=True):
        medium_count = memory_states['medium_term']['count']
        st.metric("Stored Summaries", medium_count)
        
        if medium_count > 0:
            st.markdown("**Session summaries:**")
            for summary in memory_states['medium_term']['summaries'][-2:]:
                st.markdown(f"📌 **{summary['turn_range']}**")
                st.caption(summary['summary'][:100] + "...")
        else:
            st.info("Summaries created every 5 turns")
    
    # Long-term memory
    with st.expander(" Long-term Memory (Legal DB)", expanded=True):
        long_count = memory_states['long_term']['count']
        st.metric("Retrieved Documents", long_count)
        
        if long_count > 0:
            st.markdown("**Last retrieved sections:**")
            for doc in memory_states['long_term']['last_retrieved'][:3]:
                section = doc['metadata'].get('section', 'Unknown')
                score = doc['score']
                st.markdown(f"⚖️ **{section}** (Score: {score:.2f})")
        else:
            st.info("Documents retrieved on query")
    
    # Validation score
    st.markdown("---")
    validation_score = memory_states['validation']['last_similarity']
    if validation_score > 0:
        st.markdown("###  Response Validation")
        st.progress(validation_score)
        st.caption(f"Semantic similarity: {validation_score:.2%}")
    
    # Controls
    st.markdown("---")
    if st.button(" Clear Short-term Memory"):
        st.session_state.chatbot.short_term.clear()
        st.session_state.chat_history = []
        st.success("Cleared!")
        st.rerun()
    
    if st.button(" End Session"):
        st.session_state.chatbot.end_session(st.session_state.user_id)
        st.success("Session ended! Summary saved.")
        st.rerun()

# ============================================================================ #
# MAIN CONTENT - CHAT INTERFACE
# ============================================================================ #

st.markdown('<p class="main-header">⚖️ Memory-Augmented Legal AI Assistant</p>', unsafe_allow_html=True)
st.markdown("**Ask questions about Indian Penal Code (IPC), Criminal Procedure Code (CrPC), and more!**")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        " Short-term",
        f"{memory_states['short_term']['count']} turns",
        help="Recent conversation context"
    )

with col2:
    st.metric(
        " Medium-term",
        f"{memory_states['medium_term']['count']} summaries",
        help="Session summaries (created every 5 turns)"
    )

with col3:
    st.metric(
        " Long-term",
        f"{memory_states['long_term']['count']} docs",
        help="Retrieved legal documents"
    )

with col4:
    validation_score = memory_states['validation']['last_similarity']
    st.metric(
        " Validation",
        f"{validation_score:.0%}" if validation_score > 0 else "N/A",
        help="Response accuracy score"
    )

st.markdown("---")

# ============================================================================ #
# CHAT DISPLAY
# ============================================================================ #

# Chat container
chat_container = st.container()

with chat_container:
    st.markdown("###  Conversation")
    
    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="chat-user">
                <b>👤 You:</b> {msg['user']}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
            <div class="chat-bot">
                <b> Assistant:</b> {msg['bot']}<br>
                <small style="color: #666;">
                    Intent: {msg['intent']} | 
                    Validation: {msg['validation_score']:.0%}
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Show memory sources in expander
            with st.expander(" View Memory Sources", expanded=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("** Long-term (Legal Docs):**")
                    for doc in msg.get('long_term_docs', [])[:2]:
                        st.markdown(f"- **{doc['metadata'].get('section', 'Unknown')}** (Score: {doc['score']:.2f})")
                        st.caption(doc['text'][:150] + "...")
                
                with col_b:
                    st.markdown("** Medium-term (Summaries):**")
                    summaries = msg.get('medium_term_summaries', [])
                    if summaries:
                        for summ in summaries[:2]:
                            st.caption(f"• {summ['text'][:100]}...")
                    else:
                        st.caption("No previous summaries used")
    else:
        st.info("👋 Start a conversation! Ask about legal sections, punishments, procedures, etc.")

# ============================================================================ #
# USER INPUT
# ============================================================================ #

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question:",
        value=st.session_state.get('example_query', ''),
        placeholder="Ask about IPC sections, legal procedures, punishments...",
        label_visibility="collapsed"
    )
    submit = st.form_submit_button(" Send")

# Clear example query after use
if 'example_query' in st.session_state:
    del st.session_state.example_query

if submit and user_input:
    with st.spinner(" Thinking..."):
        # Get response from chatbot
        result = st.session_state.chatbot.chat(
            query=user_input,
            user_id=st.session_state.user_id
        )
        
        # Store in chat history
        st.session_state.chat_history.append({
            'user': user_input,
            'bot': result['response'],
            'intent': result['intent'],
            'validation_score': result['validation_score'],
            'long_term_docs': result['long_term_docs'],
            'medium_term_summaries': result['medium_term_summaries']
        })
    
    st.rerun()

# ============================================================================ #
# DETAILED MEMORY ANALYSIS (Bottom Section)
# ============================================================================ #

st.markdown("---")
st.markdown("###  Detailed Memory Analysis")

tab1, tab2, tab3 = st.tabs([" Short-term Details", " Medium-term Details", " Long-term Details"])

with tab1:
    st.markdown("**Recent Conversation Turns**")
    short_turns = memory_states['short_term']['turns']
    if short_turns:
        for turn in reversed(short_turns):
            with st.expander(f"Turn {turn['turn']} - {turn['timestamp'][:19]}"):
                st.markdown(f"**User:** {turn['user']}")
                st.markdown(f"**Bot:** {turn['bot']}")
    else:
        st.info("No conversation turns yet")

with tab2:
    st.markdown("**Session Summaries (Created every 5 turns)**")
    medium_summaries = memory_states['medium_term']['summaries']
    if medium_summaries:
        for summ in reversed(medium_summaries):
            with st.expander(f"{summ['turn_range']} - {summ['timestamp'][:19]}"):
                st.markdown(summ['summary'])
    else:
        st.info("No summaries created yet. Chat for 5+ turns to see medium-term memory in action!")

with tab3:
    st.markdown("**Retrieved Legal Documents**")
    long_docs = memory_states['long_term']['last_retrieved']
    if long_docs:
        for i, doc in enumerate(long_docs, 1):
            with st.expander(f"{i}. {doc['metadata'].get('section', 'Unknown')} (Score: {doc['score']:.2f})"):
                st.markdown(f"**Category:** {doc['metadata'].get('category', 'N/A')}")
                st.markdown(f"**Content:**")
                st.text(doc['text'])
    else:
        st.info("No documents retrieved yet. Ask a question to see long-term memory retrieval!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>Memory-Augmented Legal Chatbot | Architecture: Intent → Memory Layers → RAG → Response → Guardrails</small>
</div>
""", unsafe_allow_html=True)