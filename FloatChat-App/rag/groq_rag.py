import os
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from groq import Groq
from langchain_groq import ChatGroq
from sqlalchemy import text
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGroqRAG:
    """
    Enhanced Retrieval-Augmented Generation system using Groq with direct database access
    """
    
    def __init__(self, api_key: str = None, db_manager=None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.db_manager = db_manager  # DatabaseManager instance
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable.")
        
        if not self.db_manager:
            raise ValueError("DatabaseManager instance is required.")
        
        # Initialize Groq clients
        self.groq_client = Groq(api_key=self.api_key)
        self.chat_model = ChatGroq(
            api_key=self.api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )
        
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for oceanographic data queries"""
        return """You are an expert oceanographic data analyst with direct access to ARGO float data. 

You can analyze real oceanographic measurements including:
- Temperature, salinity, pressure profiles
- Biogeochemical parameters (oxygen, nitrate, pH, chlorophyll)
- Spatial and temporal patterns
- Float trajectories and deployment data

When provided with actual data from the database:
1. Analyze the specific measurements and values
2. Provide scientifically accurate interpretations
3. Highlight interesting patterns or anomalies
4. Explain oceanographic phenomena based on the data
5. Suggest additional analyses when appropriate

Always reference the actual data values in your responses and provide context about what the measurements mean oceanographically."""

    def execute_database_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query against the database and return results"""
        try:
            with self.db_manager.engine.connect() as conn:
                result = pd.read_sql(text(sql_query), conn)
                logger.info(f"Executed database query, returned {len(result)} rows")
                return result
        except Exception as e:
            logger.error(f"Failed to execute database query: {str(e)}")
            return pd.DataFrame()

    def get_contextual_data(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant data from database based on query analysis"""
        context_data = {
            'profiles': pd.DataFrame(),
            'measurements': pd.DataFrame(),
            'statistics': {},
            'query_info': query_analysis
        }
        
        try:
            # Build database filters from query analysis
            filters = self._build_db_filters(query_analysis)
            
            # Get relevant profiles
            profiles_df = self.db_manager.get_profiles(
                limit=50,  # Reasonable limit for context
                filters=filters
            )
            
            if not profiles_df.empty:
                context_data['profiles'] = profiles_df
                
                # Get measurements for these profiles
                profile_ids = profiles_df['id'].tolist()
                all_measurements = []
                
                # Limit to first 10 profiles for performance
                for profile_id in profile_ids[:10]:
                    measurements = self.db_manager.get_measurements_by_profile(profile_id)
                    if not measurements.empty:
                        measurements['profile_id'] = profile_id
                        all_measurements.append(measurements)
                
                if all_measurements:
                    context_data['measurements'] = pd.concat(all_measurements, ignore_index=True)
                    
                    # Calculate statistics
                    context_data['statistics'] = self._calculate_statistics(
                        context_data['measurements'], 
                        query_analysis.get('parameters', [])
                    )
            
        except Exception as e:
            logger.error(f"Failed to get contextual data: {str(e)}")
        
        return context_data

    def _build_db_filters(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Convert query analysis to database filters"""
        filters = {}
        
        # Location filters
        location = query_analysis.get('location', {})
        if location:
            if 'latitude' in location and 'longitude' in location:
                # Point search with radius
                filters['latitude'] = location['latitude']
                filters['longitude'] = location['longitude']
                if 'radius_km' in location:
                    # For simplicity, convert to rough lat/lon bounds
                    radius_deg = location['radius_km'] / 111.0  # Rough conversion
                    filters['min_lat'] = location['latitude'] - radius_deg
                    filters['max_lat'] = location['latitude'] + radius_deg
                    filters['min_lon'] = location['longitude'] - radius_deg
                    filters['max_lon'] = location['longitude'] + radius_deg
            elif 'lat_range' in location:
                filters['min_lat'], filters['max_lat'] = location['lat_range']
                if 'lon_range' in location:
                    filters['min_lon'], filters['max_lon'] = location['lon_range']
        
        # Time filters
        time_range = query_analysis.get('time_range', {})
        if 'start_date' in time_range:
            filters['start_date'] = time_range['start_date']
        if 'end_date' in time_range:
            filters['end_date'] = time_range['end_date']
        
        # Float ID filters
        float_ids = query_analysis.get('float_ids', [])
        if float_ids:
            filters['float_ids'] = float_ids
        
        return filters

    def _calculate_statistics(self, measurements_df: pd.DataFrame, parameters: List[str]) -> Dict[str, Any]:
        """Calculate statistics for the measurements"""
        stats = {}
        
        if measurements_df.empty:
            return stats
        
        # Default parameters if none specified
        if not parameters:
            parameters = ['temperature', 'salinity', 'pressure']
        
        for param in parameters:
            if param in measurements_df.columns:
                param_data = measurements_df[param].dropna()
                if not param_data.empty:
                    stats[param] = {
                        'count': len(param_data),
                        'mean': float(param_data.mean()),
                        'std': float(param_data.std()),
                        'min': float(param_data.min()),
                        'max': float(param_data.max()),
                        'median': float(param_data.median())
                    }
        
        return stats

    def _format_context_for_llm(self, context_data: Dict[str, Any]) -> str:
        """Format the retrieved data for the LLM prompt"""
        if context_data['profiles'].empty:
            return "No relevant data found in the database for this query."
        
        context_text = []
        
        # Profile summary
        profiles = context_data['profiles']
        context_text.append(f"Found {len(profiles)} relevant profiles from {profiles['float_id'].nunique()} unique floats")
        
        # Geographic coverage
        if not profiles.empty:
            lat_range = f"{profiles['latitude'].min():.2f}°N to {profiles['latitude'].max():.2f}°N"
            lon_range = f"{profiles['longitude'].min():.2f}°E to {profiles['longitude'].max():.2f}°E"
            context_text.append(f"Geographic coverage: {lat_range}, {lon_range}")
        
        # Time coverage
        if 'measurement_date' in profiles.columns:
            dates = pd.to_datetime(profiles['measurement_date'])
            context_text.append(f"Time range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        
        # Measurement statistics
        if context_data['statistics']:
            context_text.append("\nMeasurement Statistics:")
            for param, stats in context_data['statistics'].items():
                context_text.append(
                    f"- {param.title()}: "
                    f"mean={stats['mean']:.2f}, "
                    f"range=[{stats['min']:.2f}, {stats['max']:.2f}], "
                    f"n={stats['count']} measurements"
                )
        
        # Sample data points
        if not context_data['measurements'].empty:
            measurements = context_data['measurements']
            context_text.append(f"\nTotal measurements: {len(measurements)} data points")
            
            # Depth range
            if 'depth' in measurements.columns:
                depth_data = measurements['depth'].dropna()
                if not depth_data.empty:
                    context_text.append(f"Depth range: {depth_data.min():.1f}m to {depth_data.max():.1f}m")
        
        return "\n".join(context_text)

    async def process_query_with_data(self, user_question: str, query_analysis: Dict[str, Any]) -> str:
        """Process query using actual database data"""
        try:
            # Get contextual data from database
            context_data = self.get_contextual_data(query_analysis)
            
            # Format context for LLM
            formatted_context = self._format_context_for_llm(context_data)
            
            # Create enhanced prompt
            prompt = f"""
Based on the following ARGO oceanographic data from our database:

{formatted_context}

User Question: "{user_question}"

Please provide a comprehensive answer that:
1. Directly addresses the user's question using the actual data
2. References specific values and measurements from the dataset
3. Explains relevant oceanographic concepts
4. Highlights any interesting patterns or findings
5. Suggests additional analyses if appropriate

Use the actual data values in your response and provide oceanographic context.
"""
            
            # Get response from Groq
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Add data summary if significant data was found
            if not context_data['profiles'].empty:
                data_summary = f"\n\n📊 **Data Summary**: Analyzed {len(context_data['profiles'])} profiles"
                if not context_data['measurements'].empty:
                    data_summary += f" with {len(context_data['measurements'])} measurements"
                answer += data_summary
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to process query with data: {str(e)}")
            return f"I encountered an error while analyzing the data: {str(e)}"

    async def process_query(self, user_question: str):
        """Process user query using enhanced RAG system with database integration - compatible with Streamlit app"""
        try:
            # Import QueryProcessor here to avoid circular imports
            # This assumes QueryProcessor is available in the session state or can be imported
            try:
                from rag.query_processor import QueryProcessor
                query_processor = QueryProcessor()
                query_analysis = query_processor.analyze_query(user_question)
            except ImportError:
                # If QueryProcessor not available, create basic query analysis
                query_analysis = {
                    'query_type': 'general_search',
                    'parameters': [],
                    'location': {},
                    'time_range': {},
                    'float_ids': []
                }
            
            # Use the existing method to process with data
            answer = await self.process_query_with_data(user_question, query_analysis)
            
            # Get contextual data for visualizations
            context_data = self.get_contextual_data(query_analysis)
            
            # Convert profiles to search results format for compatibility
            search_results = []
            if not context_data['profiles'].empty:
                for _, profile in context_data['profiles'].iterrows():
                    search_result = {
                        'profile_id': profile['id'],
                        'summary': {
                            'float_id': profile['float_id'],
                            'latitude': profile['latitude'],
                            'longitude': profile['longitude'],
                            'measurement_date': profile['measurement_date'],
                            'statistics': context_data['statistics']
                        },
                        'search_text': f"Float {profile['float_id']} at {profile['latitude']:.2f}°N, {profile['longitude']:.2f}°E on {profile['measurement_date']}",
                        'similarity_score': 0.9
                    }
                    search_results.append(search_result)
            
            return {
                'answer': answer,
                'query_analysis': query_analysis,
                'search_results': search_results,
                'relevant_data': search_results,
                'database_enhanced': True,
                'data_statistics': context_data.get('statistics', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'query_analysis': {},
                'search_results': [],
                'relevant_data': [],
                'database_enhanced': False
            }

    def generate_sql_query(self, user_question: str, database_schema: Dict[str, Any] = None) -> str:
        """Generate SQL query from natural language question"""
        try:
            schema_description = self._format_schema_description(database_schema or {})
            
            prompt = f"""
Given this database schema for ARGO oceanographic data:

{schema_description}

Generate a PostgreSQL query to answer this question: "{user_question}"

Rules:
1. Use only the tables and columns shown in the schema
2. Include appropriate WHERE clauses for filtering
3. Use JOINs when data from multiple tables is needed
4. Include ORDER BY and LIMIT clauses when appropriate
5. Handle NULL values appropriately
6. Use oceanographically meaningful constraints (e.g., reasonable temperature ranges)

Return only the SQL query without explanations:
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=512
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response to extract just the SQL
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].strip()
            
            logger.info(f"Generated SQL query for: {user_question}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Failed to generate SQL query: {str(e)}")
            return ""
    
    def _format_schema_description(self, schema: Dict[str, Any]) -> str:
        """Format database schema for prompt"""
        schema_text = "Database Tables:\n\n"
        
        tables_info = {
            'argo_profiles': {
                'description': 'Main profile information for each ARGO float deployment',
                'columns': [
                    'id (PRIMARY KEY)',
                    'float_id (VARCHAR) - ARGO float identifier',
                    'cycle_number (INTEGER) - Profile cycle number',
                    'latitude (DECIMAL) - Measurement latitude',
                    'longitude (DECIMAL) - Measurement longitude', 
                    'measurement_date (TIMESTAMP) - Date/time of measurement',
                    'platform_number (VARCHAR) - Platform identifier',
                    'data_center (VARCHAR) - Data center code'
                ]
            },
            'argo_measurements': {
                'description': 'Individual measurements at different depths',
                'columns': [
                    'id (PRIMARY KEY)',
                    'profile_id (INTEGER) - Foreign key to argo_profiles',
                    'pressure (DECIMAL) - Water pressure in decibars',
                    'temperature (DECIMAL) - Water temperature in Celsius',
                    'salinity (DECIMAL) - Practical salinity in PSU',
                    'depth (DECIMAL) - Depth in meters',
                    'oxygen (DECIMAL) - Dissolved oxygen in micromole/kg',
                    'nitrate (DECIMAL) - Nitrate in micromole/kg',
                    'ph (DECIMAL) - pH value',
                    'chlorophyll (DECIMAL) - Chlorophyll-a in mg/m3',
                    'quality_flag (INTEGER) - Data quality flag (1=good, 4=bad)'
                ]
            }
        }
        
        for table_name, table_info in tables_info.items():
            schema_text += f"Table: {table_name}\n"
            schema_text += f"Description: {table_info['description']}\n"
            schema_text += "Columns:\n"
            for column in table_info['columns']:
                schema_text += f"  - {column}\n"
            schema_text += "\n"
        
        return schema_text

    async def query(self, question: str, query_analysis: Dict[str, Any] = None) -> str:
        """Main query method that uses database data"""
        if query_analysis:
            return await self.process_query_with_data(question, query_analysis)
        else:
            # Fallback to basic query without context
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            return response.choices[0].message.content.strip()

    def answer_question_with_context(self, question: str, retrieved_data: List[Dict[str, Any]]) -> str:
        """Answer question using retrieved data as context - for compatibility"""
        try:
            # Format retrieved data for context
            context = self._format_retrieved_data(retrieved_data)
            
            prompt = f"""
Based on the following ARGO oceanographic data:

{context}

Please answer this question: "{question}"

Provide a comprehensive answer that:
1. Directly addresses the question
2. Explains relevant oceanographic concepts
3. References specific data points when available
4. Suggests additional analysis if appropriate
5. Notes any limitations or data quality considerations

Be scientifically accurate and educational in your response.
"""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for question: {question}")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to answer question with context: {str(e)}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
    
    def _format_retrieved_data(self, data: List[Dict[str, Any]]) -> str:
        """Format retrieved data for use in prompts - for compatibility"""
        if not data:
            return "No specific data found for this query."
        
        formatted_parts = []
        
        for i, item in enumerate(data[:5], 1):  # Limit to top 5 results
            part = f"Profile {i}:\n"
            
            # Profile metadata
            summary = item.get('summary', {})
            if 'float_id' in summary:
                part += f"  Float ID: {summary['float_id']}\n"
            if 'latitude' in summary and 'longitude' in summary:
                part += f"  Location: {summary['latitude']:.2f}°N, {summary['longitude']:.2f}°E\n"
            if 'measurement_date' in summary:
                part += f"  Date: {summary['measurement_date']}\n"
            
            # Statistics
            if 'statistics' in summary:
                stats = summary['statistics']
                part += "  Measurements:\n"
                for param, param_stats in stats.items():
                    if isinstance(param_stats, dict):
                        mean_val = param_stats.get('mean', 'N/A')
                        min_val = param_stats.get('min', 'N/A')
                        max_val = param_stats.get('max', 'N/A')
                        if isinstance(mean_val, (int, float)) and mean_val != 'N/A':
                            part += f"    {param.title()}: {mean_val:.2f} (range: {min_val:.2f} - {max_val:.2f})\n"
            
            # Search text summary
            if 'search_text' in item:
                part += f"  Summary: {item['search_text']}\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)

# Create alias for compatibility with existing code
GroqRAGSystem = EnhancedGroqRAG