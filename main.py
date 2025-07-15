import streamlit as st
from review_analyzer import ReviewAnalyzer # Import the custom review analysis class
from PIL import Image # For handling image data
import requests # For making HTTP requests to fetch game headers
from io import BytesIO # To work with image data in memory
import matplotlib.pyplot as plt # For plotting word clouds

def get_game_header(appid):
    """
    Fetches the header image for a given Steam game from the Steam CDN.

    Args:
        appid (str): The Steam Application ID of the game.

    Returns:
        PIL.Image.Image: A PIL Image object if the header is successfully fetched,
                         otherwise None.
    """
    try:
        # Construct the URL for the game's header image
        url = f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
        response = requests.get(url, timeout=10) # Set a timeout for the request
        # Return the image if the request was successful (status code 200), else None
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to fetch header for appid {appid}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch specific request exceptions (e.g., network issues, timeouts)
        print(f"Error fetching game header for appid {appid}: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during image processing
        print(f"An unexpected error occurred while processing game header for appid {appid}: {e}")
        return None

def display_insight_section(title, content):
    """
    Displays a section with a title and content.
    If the content indicates 'Not enough data' or is empty, a warning is shown instead.

    Args:
        title (str): The title of the insight section.
        content (str): The content/summary to display for the section.
    """
    if content and not content.startswith("Not enough"):
        st.markdown(f"### {title}") # Display title as a markdown heading
        st.markdown(content) # Display the content
    else:
        # Display a warning if content is insufficient or unavailable
        st.warning(f"{title}: {content if content else 'No data available'}")

def main():
    """
    Main function to run the Streamlit application for Steam Game Review Analysis.
    It sets up the page configuration, handles user input for game search,
    triggers review analysis, and displays the results including summaries and word clouds.
    """
    # Configure the Streamlit page layout
    st.set_page_config(layout="wide", page_title="Steam Review Insights")
    st.title("🎮 Steam Game Review Analyzer") # Main title of the application
    
    # Initialize the ReviewAnalyzer class
    analyzer = ReviewAnalyzer()
    
    # Text input for the user to search for a game
    search_query = st.text_input(
        "Search for a game:", 
        placeholder="e.g. Stardew Valley, Cyberpunk 2077",
        help="Enter the name of a game to search on Steam."
    )
    
    # Proceed only if the search query is not empty
    if search_query.strip():
        with st.spinner("Searching for games..."): # Show a spinner while searching
            games = analyzer.search_steam_games(search_query) # Call the analyzer to search for games
            
        if games:
            # Dropdown to select a game from the search results
            selected_game = st.selectbox(
                "Select a game:",
                options=games,
                format_func=lambda x: x['name'], # Display game name in the select box
                index=0, # Default to the first game in the list
                help="Choose a game from the search results to analyze its reviews."
            )
            
            # Button to trigger the review analysis
            if st.button("Analyze Reviews", help="Click to start analyzing the selected game's reviews."):
                with st.spinner("Analyzing reviews (this may take 1-2 minutes)..."): # Show a spinner during analysis
                    insights = analyzer.get_review_insights(selected_game['appid']) # Get review insights for the selected game
                    
                    # Display game header image if available
                    header = get_game_header(selected_game['appid'])
                    if header:
                        # Use 'use_container_width=True' for better responsiveness
                        st.image(header, use_container_width=True, caption=f"{selected_game['name']} Header") 
                    else:
                        st.warning("Couldn't load game header image.")
                    
                    # Check for errors from the analysis and display results
                    if 'error' in insights:
                        st.error(insights['error']) # Display analysis error
                    else:
                        # Create two columns for displaying positive and negative feedback
                        col1, col2 = st.columns(2)
                        with col1:
                            display_insight_section("👍 Positive Feedback", insights['positive'])
                        with col2:
                            display_insight_section("👎 Negative Feedback", insights['negative'])
                        
                        # Section for word clouds, separated by a divider
                        st.divider()
                        st.subheader("📊 Most Mentioned Terms")
                        
                        # Create two columns for positive and negative word clouds
                        wc_col1, wc_col2 = st.columns(2)
                        with wc_col1:
                            if insights['wordclouds']['positive']:
                                st.markdown("**Positive Reviews**")
                                plt.figure(figsize=(10, 5)) # Create a new figure for the plot
                                plt.imshow(insights['wordclouds']['positive'], interpolation='bilinear') # Display word cloud
                                plt.axis('off') # Hide axes
                                st.pyplot(plt.gcf(), use_container_width=True) # Display the plot in Streamlit
                            else:
                                st.warning("No positive word cloud available.")
                        
                        with wc_col2:
                            if insights['wordclouds']['negative']:
                                st.markdown("**Negative Reviews**")
                                plt.figure(figsize=(10, 5)) # Create a new figure for the plot
                                plt.imshow(insights['wordclouds']['negative'], interpolation='bilinear') # Display word cloud
                                plt.axis('off') # Hide axes
                                st.pyplot(plt.gcf(), use_container_width=True) # Display the plot in Streamlit
                            else:
                                st.warning("No negative word cloud available.")
        else:
            # Display warning if no games are found for the search query
            st.warning("No games found matching your search.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()