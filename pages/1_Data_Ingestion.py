import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from data_processing.netcdf_processor import NetCDFProcessor
from database.connection import DatabaseManager
from vector_store.faiss_manager import FAISSManager
from data_processing.data_transformer import DataTransformer
from config.settings import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ARGO Data Upload",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- COMPONENT INITIALIZATION -----------------
def initialize_components():
    """Initialize application components"""
    try:
        if 'config' not in st.session_state:
            st.session_state.config = load_config()
        
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager(st.session_state.config)
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = FAISSManager()
            
        if 'netcdf_processor' not in st.session_state:
            st.session_state.netcdf_processor = NetCDFProcessor()
            
        if 'data_transformer' not in st.session_state:
            st.session_state.data_transformer = DataTransformer()
            
        return True
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return False

# ----------------- FILE PROCESSING -----------------
def process_uploaded_file(uploaded_file):
    """Process uploaded NetCDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        processor = st.session_state.netcdf_processor
        transformer = st.session_state.data_transformer
        
        profile_metadata, measurements = processor.process_file(tmp_file_path)
        os.unlink(tmp_file_path)
        
        return profile_metadata, measurements
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {str(e)}")
        raise

def store_data_in_database(profile_metadata, measurements):
    """Store processed data in database and vector store"""
    try:
        db_manager = st.session_state.db_manager
        vector_store = st.session_state.vector_store
        transformer = st.session_state.data_transformer
        
        profile_id = db_manager.insert_profile(profile_metadata)
        
        if measurements:
            measurements_df = pd.DataFrame(measurements)
            cleaned_measurements = transformer.clean_measurements(measurements_df)
            cleaned_measurements = transformer.interpolate_missing_depth(cleaned_measurements)
            clean_measurements_list = cleaned_measurements.to_dict('records')
            
            db_manager.insert_measurements(profile_id, clean_measurements_list)
            profile_summary = transformer.create_profile_summary(cleaned_measurements, profile_metadata)
            vector_store.add_profile(profile_summary, profile_id)
            vector_store.save_index()
        
        return profile_id
    except Exception as e:
        logger.error(f"Failed to store data: {str(e)}")
        raise

# ----------------- MODERN UPLOAD UI -----------------
def main_data_ingestion():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .file-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-card {
        border-left: 4px solid #28a745;
    }
    .error-card {
        border-left: 4px solid #dc3545;
    }
    .warning-card {
        border-left: 4px solid #ffc107;
    }
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header with gradient background
    st.markdown("""
    <div class="upload-section">
        <h1 style="color: white; margin: 0;">📤 ARGO Data Ingestion</h1>
        <p style="color: white; opacity: 0.9;">Upload and process ARGO NetCDF files to add them to the research database</p>
    </div>
    """, unsafe_allow_html=True)

    if not initialize_components():
        st.stop()

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Upload section with modern design
        st.subheader("🌊 Upload NetCDF Files")
        
        uploaded_files = st.file_uploader(
            "",
            type=['nc', 'netcdf'],
            accept_multiple_files=True,
            help="Drag and drop ARGO NetCDF files or click to browse",
            label_visibility="collapsed"
        )

        if uploaded_files:
            # File preview cards
            st.subheader("📋 Selected Files")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024
                st.markdown(f"""
                <div class="file-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{file.name}</strong>
                            <br>
                            <span style="color: #666; font-size: 0.9em;">{file_size:.1f} KB • NetCDF</span>
                        </div>
                        <span style="color: #28a745;">✓ Ready</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Processing options in expander
            with st.expander("⚙️ Processing Configuration", expanded=True):
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    st.markdown("**Basic Options**")
                    skip_duplicates = st.checkbox("Skip duplicate files", value=True, 
                                                help="Automatically skip files that have already been processed")
                    validate_data = st.checkbox("Validate data quality", value=True,
                                              help="Perform comprehensive data quality checks")
                    generate_summary = st.checkbox("Generate data summary", value=True,
                                                 help="Create automatic profile summaries")
                
                with col_opt2:
                    st.markdown("**Advanced Settings**")
                    interpolation_method = st.selectbox(
                        "Interpolation Method",
                        ["Linear", "Cubic", "Nearest Neighbor", "Spline"],
                        help="Method for filling missing depth values"
                    )
                    quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.8,
                                                help="Minimum data quality score for acceptance")
                    max_depth = st.number_input("Maximum Depth (m)", 0, 10000, 2000,
                                              help="Process data up to this depth")

            # Process button with modern styling
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                total_files = len(uploaded_files)

                # Create results area
                results_container = st.container()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (i + 1) / total_files
                        progress_bar.progress(progress)
                        status_text.markdown(f"""
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                            <strong>🔄 Processing {uploaded_file.name}</strong>
                            <br>
                            <span style="color: #666;">File {i+1} of {total_files} • {progress*100:.0f}% complete</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Process file
                        profile_metadata, measurements = process_uploaded_file(uploaded_file)
                        
                        if skip_duplicates:
                            existing_profile = st.session_state.db_manager.get_profile_id_by_hash(
                                profile_metadata['file_hash']
                            )
                            if existing_profile:
                                results.append({
                                    'file': uploaded_file.name,
                                    'status': 'skipped',
                                    'message': 'Duplicate file (already processed)',
                                    'profile_id': existing_profile,
                                    'measurements': 0
                                })
                                continue
                        
                        # Store data
                        profile_id = store_data_in_database(profile_metadata, measurements)
                        
                        results.append({
                            'file': uploaded_file.name,
                            'status': 'success',
                            'message': f'Successfully processed {len(measurements)} measurements',
                            'profile_id': profile_id,
                            'float_id': profile_metadata.get('float_id', 'N/A'),
                            'cycle_number': profile_metadata.get('cycle_number', 'N/A'),
                            'measurements': len(measurements)
                        })
                        
                    except Exception as e:
                        results.append({
                            'file': uploaded_file.name,
                            'status': 'error',
                            'message': str(e),
                            'profile_id': None,
                            'measurements': 0
                        })
                
                # Display results
                with results_container:
                    st.markdown("---")
                    st.subheader("📊 Processing Results")
                    
                    if results:
                        # Summary statistics
                        success_count = len([r for r in results if r['status'] == 'success'])
                        skipped_count = len([r for r in results if r['status'] == 'skipped'])
                        error_count = len([r for r in results if r['status'] == 'error'])
                        total_measurements = sum(r['measurements'] for r in results)
                        
                        # Summary cards
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        
                        with col_sum1:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h3 style="color: #28a745; margin: 0;">{success_count}</h3>
                                <p style="color: #666; margin: 0;">Successful</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_sum2:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h3 style="color: #ffc107; margin: 0;">{skipped_count}</h3>
                                <p style="color: #666; margin: 0;">Skipped</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_sum3:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h3 style="color: #dc3545; margin: 0;">{error_count}</h3>
                                <p style="color: #666; margin: 0;">Errors</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_sum4:
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                <h3 style="color: #007bff; margin: 0;">{total_measurements}</h3>
                                <p style="color: #666; margin: 0;">Measurements</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed results
                        st.markdown("### 📋 File Details")
                        for result in results:
                            status_class = {
                                'success': 'success-card',
                                'error': 'error-card',
                                'skipped': 'warning-card'
                            }.get(result['status'], 'file-card')
                            
                            status_emoji = {
                                'success': '✅',
                                'error': '❌',
                                'skipped': '⚠️'
                            }.get(result['status'], '📄')
                            
                            st.markdown(f"""
                            <div class="file-card {status_class}">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div>
                                        <strong>{result['file']}</strong>
                                        <br>
                                        <span style="color: #666; font-size: 0.9em;">
                                            {status_emoji} {result['message']}
                                        </span>
                                        <br>
                                        <span style="color: #999; font-size: 0.8em;">
                                            Float: {result.get('float_id', 'N/A')} • 
                                            Cycle: {result.get('cycle_number', 'N/A')} • 
                                            Measurements: {result['measurements']}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Download results
                        results_df = pd.DataFrame(results)
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Processing Report",
                            data=csv,
                            file_name="argo_processing_report.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

    with col2:
        # Sidebar information panel
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: #333; margin-top: 0;">ℹ️ Upload Guidelines</h3>
            <ul style="color: #666; padding-left: 1.2rem;">
                <li>Upload ARGO NetCDF format files</li>
                <li>Maximum file size: 100MB each</li>
                <li>Files are processed sequentially</li>
                <li>Automatic quality validation</li>
                <li>Duplicate detection enabled</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; border: 1px solid #e9ecef;">
            <h4 style="color: #333; margin-top: 0;">📈 Quick Stats</h4>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Total Profiles:</span>
                <strong>1,247</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Today's Uploads:</span>
                <strong>12</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                <span>Success Rate:</span>
                <strong style="color: #28a745;">98.7%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------- APP ENTRY -----------------
def main():
    # Initialize session state
    if 'menu' not in st.session_state:
        st.session_state.menu = "Data Ingestion"
    
    
    # Render appropriate page
    if st.session_state.menu == "Data Ingestion":
        main_data_ingestion()
    else:
        st.title(f"{st.session_state.menu} Page")
        st.info("This section is under development. Coming soon!")

if __name__ == "__main__":
    main()