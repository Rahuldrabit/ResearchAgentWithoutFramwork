import streamlit as st
import pandas as pd
import time
import json

# Try to prevent PyTorch module initialization issues with Streamlit
try:
    import torch
    # Set PyTorch settings that might help avoid conflicts
    torch.set_grad_enabled(False)
    print("PyTorch imported successfully and gradient tracking disabled")
except (ImportError, RuntimeError) as e:
    print(f"Note: PyTorch not imported or had an issue: {e}")
    # This is fine, we don't directly use PyTorch in this UI

from tools.mcp_client import MCPClient

# Helper function for Streamlit version compatibility
def rerun():
    """Rerun the Streamlit app, compatible with different Streamlit versions"""
    try:
        # New Streamlit versions
        st.rerun()
    except AttributeError:
        try:
            # Older Streamlit versions
            st.experimental_rerun()
        except AttributeError:
            # Fallback for very old versions
            st.empty()
            raise Exception("Could not rerun the app. Please refresh the page manually.")

# Initialize the client
client = MCPClient(base_url=st.sidebar.text_input("Server URL", value="http://localhost:8000"))

# Set up the sidebar navigation
st.sidebar.title("Research Assistant")
page = st.sidebar.radio("Navigation", ["Search", "Chat", "Research Pipeline", "QA Analytics"])

# Session state initialization
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(int(time.time()))
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "search_results" not in st.session_state:
    st.session_state.search_results = []
    
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None

# Search page
if page == "Search":
    st.title("Search Research Papers")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Enter search query:")
    
    with col2:
        limit = st.number_input("Number of results:", min_value=1, max_value=20, value=5)
    
    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                results = client.search_papers(search_query, limit)
                st.session_state.search_results = results
                st.success(f"Found {len(results)} papers")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.search_results:
        st.subheader("Search Results")
        for i, paper in enumerate(st.session_state.search_results, 1):
            with st.expander(f"{i}. {paper['title']}"):
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Keywords:** {', '.join(paper['keywords'])}")
                st.write("**Abstract:**")
                st.write(paper['abstract'])
                
                if st.button(f"Explain Paper #{i}", key=f"explain_{i}"):
                    explanation_container = st.container()
                    with explanation_container:
                        with st.spinner("Generating explanation..."):
                            try:
                                result = client.explain_paper(paper)
                                
                                if result.get("status") == "completed":
                                    st.success("✅ Explanation generated!")
                                    
                                    # Display explanation with improved visibility
                                    if "result" in result and "explanation" in result["result"]:
                                        explanation_text = result["result"]["explanation"]
                                        
                                        # Create a visible container for the explanation
                                        st.subheader("📝 Explanation:")
                                        explanation_box = st.container()
                                        with explanation_box:
                                            st.markdown("---")
                                            # Use plain text display for reliability
                                            st.text_area("Paper Explanation", explanation_text, 
                                                         height=300, key=f"explanation_text_{i}")
                                            st.markdown("---")
                                        
                                        # Display quality score with error handling
                                        quality_score = result.get("quality_score", 0)
                                        if quality_score is not None:
                                            try:
                                                st.write(f"**Quality Score:** {float(quality_score):.2f}/1.0")
                                            except (ValueError, TypeError):
                                                st.write(f"**Quality Score:** {quality_score}/1.0")
                                        else:
                                            st.write("**Quality Score:** Not available")
                                        
                                        # Debug toggle for advanced users
                                        with st.expander("Show Debug Information"):
                                            st.write("**Raw Response:**")
                                            st.json(result)
                                    else:
                                        st.error("❌ No explanation text found in the response")
                                        st.write("Response structure was:")
                                        if "result" in result:
                                            st.write(f"Result keys: {list(result['result'].keys())}")
                                        st.json(result)
                                else:
                                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                                    st.write("Full response:")
                                    st.json(result)
                            except RuntimeError as re:
                                if "Tried to instantiate class '__path__._path'" in str(re):
                                    st.error("Error with PyTorch integration. This is a known issue with Streamlit's file watcher and PyTorch. Try restarting the application or disabling PyTorch if it's not needed.")
                                    # Log the error for debugging
                                    print(f"PyTorch RuntimeError: {str(re)}")
                                else:
                                    st.error(f"Runtime Error: {str(re)}")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

# Chat page
elif page == "Chat":
    st.title("Research Assistant Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
            if message.get("sources"):
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    title = source.get("title", "Unknown")
                    authors = source.get("authors", [])
                    authors_text = ", ".join(authors[:2])
                    if len(authors) > 2:
                        authors_text += " et al."
                    score = source.get("relevance_score", 0)
                    st.markdown(f"{i}. {title} by {authors_text} (Relevance: {score:.2f})")
    
    # New question input
    question = st.text_input("Ask a question about research papers:")
    max_papers = st.slider("Maximum papers to search:", min_value=1, max_value=10, value=3)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Send"):
            if question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                with st.spinner("Searching database and generating answer..."):
                    try:
                        result = client.ask_chat_question(
                            question=question,
                            session_id=st.session_state.chat_session_id,
                            max_context_papers=max_papers
                        )
                        
                        # Add assistant message to history with sources
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": result["answer"],
                            "sources": result["sources"]
                        })
                        
                        # Use the version-compatible rerun function
                        rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("New Conversation"):
            st.session_state.chat_session_id = str(int(time.time()))
            st.session_state.chat_history = []
            # Use the version-compatible rerun function
            rerun()

# Research Pipeline
elif page == "Research Pipeline":
    st.title("Research Pipeline")
    
    query = st.text_input("Research Topic:")
    num_papers = st.slider("Number of papers to retrieve:", min_value=1, max_value=10, value=3)
    
    if st.button("Start Research Pipeline"):
        with st.spinner("Running research pipeline... This may take some time."):
            try:
                # Step 1: Search for papers
                task_result = client.create_research_task(query, num_papers)
                
                if task_result.get("status") == "completed" and task_result.get("result"):
                    papers = task_result["result"]
                    
                    # Display papers
                    st.subheader(f"Found {len(papers)} papers on {query}")
                    
                    # Step 2: Process each paper
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, paper in enumerate(papers):
                        st.write(f"Processing paper: {paper['title']}")
                        
                        try:
                            # Get explanation with QA
                            explanation_result = client.explain_paper(paper)
                            
                            if explanation_result.get("status") == "completed":
                                results.append({
                                    "paper": paper,
                                    "explanation": explanation_result["result"],
                                    "quality_score": explanation_result.get("quality_score", 0)
                                })
                            else:
                                st.warning(f"Failed to process paper: {explanation_result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.warning(f"Error processing paper: {str(e)}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(papers))
                    
                    # Save results to session state
                    st.session_state.pipeline_results = results
                    st.success("Research pipeline completed!")
                    
                else:
                    st.error(f"Error in research task: {task_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display pipeline results if available
    if st.session_state.pipeline_results:
        st.subheader("Research Results")
        
        for i, result in enumerate(st.session_state.pipeline_results, 1):
            paper = result["paper"]
            explanation = result["explanation"]
            
            with st.expander(f"{i}. {paper['title']}"):
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Keywords:** {', '.join(paper.get('keywords', []))}")
                st.write(f"**Quality Score:** {result.get('quality_score', 0):.2f}/1.0")
                
                st.subheader("Abstract")
                st.write(paper['abstract'])
                
                st.subheader("Explanation")
                st.write(explanation["explanation"])
                
                if explanation.get("metadata") and explanation["metadata"].get("evaluation_metrics"):
                    metrics = explanation["metadata"]["evaluation_metrics"]
                    st.subheader("Evaluation Metrics")
                    
                    metrics_df = pd.DataFrame({
                        "Metric": ["Factual Accuracy", "Comprehensiveness", "Clarity", "Conciseness"],
                        "Score": [
                            metrics.get("factual_accuracy", 0),
                            metrics.get("comprehensiveness", 0),
                            metrics.get("clarity", 0),
                            metrics.get("conciseness", 0)
                        ]
                    })
                    
                    st.dataframe(metrics_df)

# QA Analytics page
elif page == "QA Analytics":
    st.title("Quality Assurance Analytics")
    
    if st.button("Get QA Analytics"):
        with st.spinner("Fetching QA analytics..."):
            try:
                analytics = client.get_qa_analytics()
                
                if analytics and "insights" in analytics:
                    insights = analytics["insights"]
                    
                    # Display metrics
                    if "average_metrics" in insights:
                        st.subheader("Average Quality Metrics")
                        metrics = insights["average_metrics"]
                        
                        metrics_df = pd.DataFrame({
                            "Metric": ["Overall Quality", "Factual Accuracy", "Comprehensiveness", "Clarity", "Conciseness"],
                            "Score": [
                                metrics.get("overall_quality", 0),
                                metrics.get("factual_accuracy", 0),
                                metrics.get("comprehensiveness", 0),
                                metrics.get("clarity", 0),
                                metrics.get("conciseness", 0)
                            ]
                        })
                        
                        st.dataframe(metrics_df)
                        
                        # Create a basic bar chart
                        st.bar_chart(metrics_df.set_index("Metric"))
                    
                    # Display top issues
                    if "top_issues" in insights and insights["top_issues"]:
                        st.subheader("Top Issues")
                        issues_df = pd.DataFrame(insights["top_issues"])
                        st.dataframe(issues_df)
                    
                    # Display hallucinations
                    if "top_hallucinations" in insights and insights["top_hallucinations"]:
                        st.subheader("Common Hallucinations")
                        hallucinations_df = pd.DataFrame(insights["top_hallucinations"])
                        st.dataframe(hallucinations_df)
                    
                    # Display improvement suggestions
                    if "improvement_suggestions" in insights and insights["improvement_suggestions"]:
                        st.subheader("Improvement Suggestions")
                        for suggestion in insights["improvement_suggestions"]:
                            st.markdown(f"- {suggestion}")
                    
                    st.success(f"Analytics based on {insights.get('total_evaluations', 0)} evaluations")
                else:
                    st.warning("No analytics data available")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This web interface connects to your MCP server to provide a user-friendly way "
    "to interact with your multi-agent research system."
)
