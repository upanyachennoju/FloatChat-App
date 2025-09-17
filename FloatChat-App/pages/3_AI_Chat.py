import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from database.connection import DatabaseManager
from vector_store.faiss_manager import FAISSManager
from rag.groq_rag import GroqRAGSystem
from rag.query_processor import QueryProcessor
from mcp.integration import MCPEnhancedRAG, MCPToolHelper
from visualization.plots import OceanographicPlots
from visualization.maps import OceanographicMaps
from config.settings import load_config
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .header-container {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .settings-button {
        position: absolute;
        top: 0;
        right: 0;
        background-color: #f8f9fa;
        border: 1px solid #dadce0;
        border-radius: 20px;
        padding: 8px 16px;
        cursor: pointer;
        font-size: 14px;
        color: #5f6368;
        transition: all 0.2s ease;
    }
    .settings-button:hover {
        background-color: #e8f0fe;
        border-color: #1a73e8;
        color: #1a73e8;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #1a73e8;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
    }
    .bot-message {
        background-color: #e8f0fe;
        color: #1a1a1a;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: 0;
        margin-right: auto;
    }
    .message-timestamp {
        font-size: 0.75rem;
        color: #5f6368;
        margin-top: 4px;
    }
    .quick-question-btn {
        border: 1px solid #dadce0;
        border-radius: 18px;
        padding: 6px 12px;
        margin: 4px;
        font-size: 0.8rem;
        background-color: white;
        color: #1a73e8;
        cursor: pointer;
    }
    .quick-question-btn:hover {
        background-color: #f8f9fa;
        border-color: #1a73e8;
    }
    .chat-input-container {
        display: flex;
        gap: 8px;
        margin-top: 16px;
    }
    .stChatInput {
        flex: 1;
    }
    .send-button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }
    .send-button:disabled {
        background-color: #dadce0;
        cursor: not-allowed;
    }
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 8px;
    }
    .bot-avatar {
        background-color: #1a73e8;
        color: white;
    }
    .user-avatar {
        background-color: #5f6368;
        color: white;
    }
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .user-message-container {
        display: flex;
        align-items: flex-start;
        flex-direction: row-reverse;
        margin-bottom: 16px;
    }
    .message-content {
        flex: 1;
    }
    .settings-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    .settings-content {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        max-width: 500px;
        width: 90%;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AI Assistant - ARGO Platform",
    page_icon="🤖",
    layout="wide"
)

def initialize_components():
    """Initialize application components"""
    try:
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
        
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(st.session_state.config)
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = FAISSManager()
        
        if 'query_processor' not in st.session_state:
            st.session_state.query_processor = QueryProcessor()
        
        if 'plotter' not in st.session_state:
            st.session_state.plotter = OceanographicPlots()
            
        if 'mapper' not in st.session_state:
            st.session_state.mapper = OceanographicMaps()
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize settings modal state
        if 'show_settings' not in st.session_state:
            st.session_state.show_settings = False
        
        # Initialize RAG system - this should be done AFTER db_manager is created
        if 'rag_system' not in st.session_state:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                st.error("Groq API key not found. Please set GROQ_API_KEY environment variable.")
                return False
            
            # Try to use enhanced database-connected version first
            try:
                from rag.groq_rag import EnhancedGroqRAG
                # Pass the already initialized db_manager
                st.session_state.rag_system = EnhancedGroqRAG(
                    api_key=api_key, 
                    db_manager=st.session_state.db_manager  # This should now exist
                )
                st.session_state.using_enhanced_rag = True
                logger.info("Using Enhanced GroqRAG with database integration")
            except ImportError as ie:
                # Enhanced system not available, use fallback
                logger.warning(f"Enhanced GroqRAG not available: {ie}")
                groq_rag = GroqRAGSystem(api_key)
                st.session_state.rag_system = MCPEnhancedRAG(groq_rag)
                st.session_state.using_enhanced_rag = False
                logger.info("Using original MCP Enhanced RAG system")
            except Exception as e:
                # Other error with enhanced system, use fallback
                logger.error(f"Failed to initialize Enhanced GroqRAG: {e}")
                groq_rag = GroqRAGSystem(api_key)
                st.session_state.rag_system = MCPEnhancedRAG(groq_rag)
                st.session_state.using_enhanced_rag = False
                logger.info("Using original MCP Enhanced RAG system as fallback")
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        logger.error(f"Component initialization error: {e}")
        return False

async def process_query(user_question):
    """Process user query using MCP Enhanced RAG system"""
    try:
        # Use MCP Enhanced RAG for processing
        answer = await st.session_state.rag_system.process_query(user_question)
        
        # Extract only text response
        if isinstance(answer, dict):
            final_answer = answer.get("answer") or answer.get("content") or str(answer)
        else:
            final_answer = str(answer)

        # Analyze the query for visualization purposes
        query_analysis = st.session_state.query_processor.analyze_query(user_question)
        
        # Search vector database for additional context if needed
        try:
            search_results = st.session_state.vector_store.search(user_question, k=5)
        except:
            search_results = []
        
        return {
            'answer': final_answer,
            'query_analysis': query_analysis,
            'search_results': search_results,
            'relevant_data': search_results,
            'mcp_enhanced': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        return {
            'answer': f"I encountered an error while processing your question: {str(e)}",
            'query_analysis': {},
            'search_results': [],
            'relevant_data': [],
            'mcp_enhanced': False
        }

def generate_visualizations(query_analysis, search_results):
    """Generate appropriate visualizations based on query"""
    visualizations = []
    
    try:
        if not search_results:
            return visualizations
        
        # Extract profile IDs from search results
        profile_ids = [result.get('profile_id') for result in search_results if result.get('profile_id')]
        
        if not profile_ids:
            return visualizations
        
        # Get profile and measurement data
        profiles_list = []
        measurements_list = []
        
        for profile_id in profile_ids[:10]:  # Limit to first 10 for performance
            try:
                # Get profile info
                profile_data = st.session_state.db_manager.get_profiles(
                    filters={'profile_ids': [profile_id]}, limit=1
                )
                if not profile_data.empty:
                    profiles_list.append(profile_data)
                
                # Get measurements
                measurements = st.session_state.db_manager.get_measurements_by_profile(profile_id)
                if not measurements.empty:
                    measurements['profile_id'] = profile_id
                    measurements_list.append(measurements)
            except Exception as e:
                logger.warning(f"Failed to get data for profile {profile_id}: {str(e)}")
                continue
        
        if profiles_list:
            all_profiles = pd.concat(profiles_list, ignore_index=True)
            
            # Geographic visualization
            if query_analysis.get('query_type') in ['location_search', 'general_search']:
                map_viz = st.session_state.mapper.create_float_trajectory_map(all_profiles)
                visualizations.append({
                    'type': 'map',
                    'title': 'Float Locations',
                    'content': map_viz
                })
        
        if measurements_list:
            all_measurements = pd.concat(measurements_list, ignore_index=True)
            
            # Parameter-specific visualizations
            parameters = query_analysis.get('parameters', [])
            
            if 'temperature' in parameters or 'salinity' in parameters:
                # T-S diagram if both are available
                if 'temperature' in all_measurements.columns and 'salinity' in all_measurements.columns:
                    ts_plot = st.session_state.plotter.create_ts_diagram(all_measurements)
                    visualizations.append({
                        'type': 'plot',
                        'title': 'Temperature-Salinity Diagram',
                        'content': ts_plot
                    })
            
            # Depth profiles for requested parameters
            available_params = [p for p in parameters 
                              if p in all_measurements.columns and all_measurements[p].notna().any()]
            
            if available_params:
                depth_profile = st.session_state.plotter.create_depth_profile(
                    all_measurements, available_params[:3], "Query Results - Depth Profiles"
                )
                visualizations.append({
                    'type': 'plot',
                    'title': 'Depth Profiles',
                    'content': depth_profile
                })
            
            # Time series if temporal analysis
            if query_analysis.get('query_type') == 'temporal_analysis' and profiles_list:
                for param in available_params[:2]:  # Limit to 2 parameters
                    if param in all_measurements.columns:
                        # Create time series data
                        time_series_data = []
                        for measurements_df in measurements_list:
                            if param in measurements_df.columns:
                                profile_id = measurements_df['profile_id'].iloc[0]
                                profile_info = all_profiles[all_profiles['id'] == profile_id]
                                if not profile_info.empty:
                                    mean_value = measurements_df[param].mean()
                                    if not pd.isna(mean_value):
                                        time_series_data.append({
                                            'measurement_date': profile_info['measurement_date'].iloc[0],
                                            param: mean_value
                                        })
                        
                        if time_series_data:
                            time_series_df = pd.DataFrame(time_series_data)
                            ts_plot = st.session_state.plotter.create_time_series(time_series_df, param)
                            visualizations.append({
                                'type': 'plot',
                                'title': f'{param.title()} Time Series',
                                'content': ts_plot
                            })
    
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {str(e)}")
    
    return visualizations

def display_chat_message(role, content, timestamp=None):
    """Display a chat message with proper formatting"""
    if timestamp is None:
        timestamp = datetime.now()
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message-container">
            <div class="avatar user-avatar">U</div>
            <div class="message-content">
                <div class="user-message">{content}</div>
                <div class="message-timestamp" style="text-align: right;">{timestamp.strftime('%H:%M:%S')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-container">
            <div class="avatar bot-avatar">A</div>
            <div class="message-content">
                <div class="bot-message">{content}</div>
                <div class="message-timestamp">{timestamp.strftime('%H:%M:%S')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_settings_modal():
    """Display settings modal"""
    with st.container():
        st.markdown("### ⚙️ Chat Settings")
        
        # Chat statistics
        if st.session_state.chat_history:
            user_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
            ai_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'assistant'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Messages", user_messages)
            with col2:
                st.metric("AI Responses", ai_messages)
        else:
            st.info("No chat history yet. Start by asking a question!")
        
        st.divider()
        
        # Chat management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("💾 Export Chat", type="secondary", use_container_width=True):
                if st.session_state.chat_history:
                    chat_export = []
                    for msg in st.session_state.chat_history:
                        chat_export.append(f"**{msg['role'].title()}** ({msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})")
                        chat_export.append(msg['content'])
                        chat_export.append("")
                    
                    chat_text = "\n".join(chat_export)
                    st.download_button(
                        "📥 Download Chat History",
                        chat_text,
                        file_name=f"argo_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.warning("No chat history to export.")
        
        st.divider()
        
        # Close button
        if st.button("✖️ Close Settings", type="primary", use_container_width=True):
            st.session_state.show_settings = False
            st.rerun()

def main():
    """Main AI chat interface"""
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Header with settings button
    st.markdown("""
    <div class="header-container">
        <h1 class="main-header">🤖 AI Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Settings button in top right
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("⚙️ Settings", key="settings_btn"):
            st.session_state.show_settings = True
            st.rerun()
    
    # Show settings modal if toggled
    if st.session_state.show_settings:
        with st.expander("⚙️ Settings", expanded=True):
            show_settings_modal()
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #5f6368;'>
        Ask questions about your oceanographic data and get AI-powered insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
        <div style="width: 24px; height: 24px; border-radius: 50%; background-color: #1a73e8; 
                    display: flex; align-items: center; justify-content: center; color: white;">A</div>
        <h3 style="margin: 0; color: #1a73e8;">FLOATCHAT Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    chat_messages_container = st.container()
    
    with chat_messages_container:
        for message in st.session_state.chat_history:
            display_chat_message(
                message['role'], 
                message['content'], 
                message.get('timestamp')
            )
    
    # Quick suggestions
    st.markdown("""
    <div style="margin: 16px 0;">
        <p style="font-size: 0.9rem; color: #5f6368; margin-bottom: 8px;">Quick questions:</p>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
    """, unsafe_allow_html=True)
    
    quick_questions = [
        "What's the average temperature in our dataset?",
        "Show me pollution levels by depth",
        "Which AI model should I use for coral reef analysis?",
        "Analyze current flow patterns"
    ]
    
    cols = st.columns(4)
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.user_input = question
                st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your oceanographic data...")
    
    # Handle example query selection
    if hasattr(st.session_state, 'user_input'):
        user_input = st.session_state.user_input
        delattr(st.session_state, 'user_input')
    
    if user_input:
        # Display user message
        timestamp = datetime.now()
        display_chat_message("user", user_input, timestamp)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': timestamp
        })
        
        # Process query and generate response
        with st.spinner("🤔 Analyzing with MCP tools..."):
            try:
                # Process the query with async support
                response_data = asyncio.run(process_query(user_input))
                
                # Display AI response
                ai_timestamp = datetime.now()
                display_chat_message("assistant", response_data['answer'], ai_timestamp)
                
                # Add to chat history with MCP indicator
                mcp_indicator = " 🚀" if response_data.get('mcp_enhanced') else ""
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_data['answer'] + mcp_indicator,
                    'timestamp': ai_timestamp
                })
                
                # Generate and display visualizations
                if response_data['search_results']:
                    st.subheader("📊 Related Visualizations")
                    
                    visualizations = generate_visualizations(
                        response_data['query_analysis'],
                        response_data['search_results']
                    )
                    
                    if visualizations:
                        # Display visualizations in tabs or columns
                        if len(visualizations) == 1:
                            viz = visualizations[0]
                            st.subheader(viz['title'])
                            if viz['type'] == 'plot':
                                st.plotly_chart(viz['content'], use_container_width=True)
                            elif viz['type'] == 'map':
                                st.components.v1.html(viz['content']._repr_html_(), height=500)
                        
                        elif len(visualizations) > 1:
                            # Create tabs for multiple visualizations
                            tab_names = [viz['title'] for viz in visualizations]
                            tabs = st.tabs(tab_names)
                            
                            for tab, viz in zip(tabs, visualizations):
                                with tab:
                                    if viz['type'] == 'plot':
                                        st.plotly_chart(viz['content'], use_container_width=True)
                                    elif viz['type'] == 'map':
                                        st.components.v1.html(viz['content']._repr_html_(), height=500)
                
                # Show relevant data sources
                if response_data['search_results']:
                    with st.expander("🔍 Data Sources Used"):
                        st.write(f"Found {len(response_data['search_results'])} relevant profiles:")
                        
                        for i, result in enumerate(response_data['search_results'][:5], 1):
                            summary = result.get('summary', {})
                            st.write(f"**{i}.** Float {summary.get('float_id', 'Unknown')} - "
                                   f"Similarity: {result.get('similarity_score', 0):.3f}")
                            if 'search_text' in result:
                                st.caption(result['search_text'][:200] + "...")
                
                # Query analysis details
                if response_data['query_analysis']:
                    with st.expander("🧠 Query Analysis"):
                        analysis = response_data['query_analysis']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Query Type:** {analysis.get('query_type', 'unknown')}")
                            st.write(f"**Parameters:** {', '.join(analysis.get('parameters', []))}")
                        
                        with col2:
                            if analysis.get('location'):
                                st.write(f"**Location:** {analysis['location']}")
                            if analysis.get('time_range'):
                                st.write(f"**Time Range:** {analysis['time_range']}")
                
            except Exception as e:
                error_msg = f"I encountered an error while processing your question: {str(e)}"
                display_chat_message("assistant", error_msg, datetime.now())
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now()
                })
        
        # Rerun to update the display
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container
    
    # Sidebar with example queries and tips
    with st.sidebar:
        st.subheader("🚀 MCP Enhanced AI")
        st.success("Now powered by Model Context Protocol (MCP) for advanced oceanographic analysis!")
        
        # MCP Tools section
        with st.expander("🛠️ Available Tools"):
            tool_descriptions = MCPToolHelper.get_tool_descriptions()
            for tool_name, description in tool_descriptions.items():
                st.write(f"**{tool_name.replace('_', ' ').title()}:** {description}")
        
        st.subheader("💡 Example Questions")
        
        example_queries = [
            "Show me temperature profiles in the Arabian Sea",
            "What are the salinity measurements near 20°N, 65°E?",
            "Compare oxygen levels in the Indian Ocean over the last year",
            "Find profiles with temperature greater than 25°C",
            "Show me data from float 2902746",
            "What is the average salinity at 500m depth?",
            "Explain mixed layer depth",
            "Show me BGC parameters in the equatorial region",
            "Analyze profiles between latitude 10 and 20",
            "Calculate water density for profile 12345",
            "Get trajectory for float 2902746",
            "Search for high temperature anomalies"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.user_input = query
                st.rerun()
        
        st.subheader("🔧 Query Tips")
        st.markdown("""
        **Location formats:**
        - "near 20°N, 65°E"
        - "in the Arabian Sea"
        - "within 100km of coordinates"
        
        **Parameter names:**
        - temperature, temp
        - salinity, salt
        - oxygen, dissolved oxygen
        - nitrate, nitrogen
        - pH, acidity
        - chlorophyll, chla
        
        **Time expressions:**
        - "in March 2023"
        - "last 6 months"
        - "between 2020 and 2022"
        
        **Comparisons:**
        - "compare X and Y"
        - "temperature vs depth"
        - "before and after"
        """)
    
    

if __name__ == "__main__":
    main()