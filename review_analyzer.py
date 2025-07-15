import requests
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import ollama
import matplotlib.pyplot as plt
from collections import Counter
import re

class ReviewAnalyzer:
    """
    A class to analyze Steam game reviews, classify their sentiment,
    extract key features, and generate word clouds for positive and negative feedback.
    """
    def __init__(self):
        """
        Initializes the ReviewAnalyzer, setting up NLTK components
        like stopwords and a lemmatizer.
        """
        # Download necessary NLTK data if not already present
        self._initialize_nltk()
        # Load English stopwords, falling back to NLTK's default if external fetch fails
        self.stopwords = self._load_stopwords()
        # Initialize the WordNet Lemmatizer for text normalization
        self.wnl = WordNetLemmatizer()
        
    def _initialize_nltk(self):
        """
        Downloads essential NLTK data packages ('punkt', 'wordnet', 'stopwords')
        required for text processing, quietly to avoid verbose output.
        """
        nltk.download('punkt', quiet=True)    # For tokenization
        nltk.download('wordnet', quiet=True)  # For lemmatization
        nltk.download('stopwords', quiet=True) # For common word filtering

    def _load_stopwords(self):
        """
        Attempts to load a comprehensive list of stopwords from a Gist URL.
        If the download fails (e.g., network error), it falls back to
        NLTK's default English stopwords.
        """
        try:
            # Fetch stopwords from a public Gist for a more extensive list
            stopwords_list = requests.get(
                "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt",
                timeout=10 # Set a timeout for the request
            ).content
            return set(stopwords_list.decode().splitlines())
        except requests.exceptions.RequestException:
            # Fallback to NLTK's built-in stopwords if fetching from URL fails
            from nltk.corpus import stopwords
            print("Warning: Could not fetch external stopwords. Using NLTK's default.")
            return set(stopwords.words('english'))
        except Exception as e:
            # Catch other potential errors during loading
            print(f"An unexpected error occurred while loading stopwords: {e}")
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))

    def search_steam_games(self, query):
        """
        Searches the Steam API for games matching a given query.
        Returns a sorted list of game dictionaries (appid, name).
        Includes enhanced error handling for API requests.

        Args:
            query (str): The search term for the game.

        Returns:
            list[dict]: A list of dictionaries, each containing 'appid' and 'name'
                        of matching games, sorted by name length. Returns an empty
                        list on error or no results.
        """
        url = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
        try:
            response = requests.get(url, timeout=10) # Set a timeout for the request
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            apps = response.json().get('applist', {}).get('apps', [])
            # Filter games where the query is in the game name (case-insensitive)
            # Sort by name length to prioritize more exact matches
            return sorted(
                [app for app in apps if query.lower() in app['name'].lower()],
                key=lambda x: len(x['name'])
            )[:20] # Limit to top 20 results
        except requests.exceptions.RequestException as e:
            print(f"Search error during API request: {e}")
        except ValueError as e:
            print(f"Search error: Could not decode JSON response: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during game search: {e}")
        return []

    def get_review_insights(self, appid, max_reviews=150):
        """
        Fetches, classifies, and summarizes reviews for a given Steam game (by appid).
        Generates positive and negative feedback summaries and corresponding word clouds.

        Args:
            appid (str): The Steam application ID of the game.
            max_reviews (int): The maximum number of reviews to fetch and analyze.

        Returns:
            dict: A dictionary containing 'positive' and 'negative' insights,
                  'wordclouds' (PIL Image objects), and an 'error' message if applicable.
        """
        try:
            reviews = self._fetch_reviews(appid, max_reviews)
            # Check if enough reviews were found for meaningful analysis
            if not reviews or len(reviews) < 10:
                return {
                    'error': f"Only {len(reviews) if reviews else 0} reviews found. Need at least 10 for analysis.",
                    'positive': "Not enough data for positive insights",
                    'negative': "Not enough data for negative insights",
                    'wordclouds': {'positive': None, 'negative': None}
                }

            # Classify reviews into positive and negative sets
            positive, negative = self._classify_reviews(reviews)
            
            insights = {
                # Generate summary for positive reviews if sufficient data, otherwise provide a default message
                'positive': self._generate_quality_summary(positive, 'positive') if len(positive) >= 5 
                            else "Not enough consistently positive reviews to identify key points.",
                # Generate summary for negative reviews if sufficient data, otherwise provide a default message
                'negative': self._generate_quality_summary(negative, 'negative') if len(negative) >= 5 
                            else "Not enough consistently negative reviews to identify key points.",
                # Generate word clouds for positive and negative reviews
                'wordclouds': {
                    'positive': self._create_wordcloud(positive, 'Greens', 'positive') if positive else None,
                    'negative': self._create_wordcloud(negative, 'Reds', 'negative') if negative else None
                }
            }
            return insights
            
        except Exception as e:
            print(f"Analysis error for appid {appid}: {e}")
            return {
                'error': f"Review analysis failed due to an unexpected error: {e}. Please try again later.",
                'positive': "Analysis unavailable",
                'negative': "Analysis unavailable",
                'wordclouds': {'positive': None, 'negative': None}
            }

    def _generate_quality_summary(self, texts, sentiment):
        """
        Generates a concise, feature-focused summary from a list of review texts
        using the Ollama language model. The prompt is carefully engineered to
        ensure the output is formatted as single, specific sentences for each feature.

        Args:
            texts (list[str]): A list of review texts (already lemmatized).
            sentiment (str): The sentiment type ('positive' or 'negative').

        Returns:
            str: A formatted string of bulleted sentences describing key features,
                 or a fallback message if not enough texts are provided or an error occurs.
        """
        if not texts or len(texts) < 5:
            return f"Not enough {sentiment} reviews to identify key points."
        
        # Combine texts into a single string, truncating to avoid exceeding model context limits
        combined = " ".join(texts)[:4000] # Limit input text to ~4000 characters

        # Detailed prompt engineering to guide the LLM to specific output format
        prompt = f"""Analyze these {sentiment} game reviews and list 3-5 SPECIFIC FEATURES players mention.
                Format each point as a single, concise sentence directly.
                
                Rules for output:
                1. Be extremely specific - name actual game features (e.g., 'career mode', 'gunplay', 'skill tree', 'graphics').
                2. Each point must be a single, complete sentence.
                3. Do NOT include phrases like 'mentioned X times', '(X)', 'review X', or any similar counts/references.
                4. Do NOT include direct quotes or examples from actual reviews.
                5. Never start a point with "players say", "reviewers note", or similar introductory phrases or summary sentences.
                6. Use natural gaming terms.
                7. Avoid generic words like 'good' or 'bad' unless paired with specifics, or if the opinion is clear from context.
                8. Ensure each bullet point is a standalone, concise summary of a feature and its opinion.
                
                Good Examples (follow this style):
                - The career mode has great depth.
                - The gunplay feels satisfying.
                - Multiplayer servers are often unstable.
                - The story lacks depth.
                - The perk tree is missing.
                
                Bad Examples (DO NOT include this style):
                - Players like the game.
                - Many enjoy the good gameplay.
                - Gameplay feel chunky (mentioned in 2 reviews).
                - "Feel chunky, vehicle handling much worse" (review 2).
                - Note: Some reviews may mention...
                - Summary of positive reviews: ...
                - Here are some features: ...
                - Feature: Weapon variety
                  Opinion: "The weapon selection is great..."
                
                Reviews to analyze: {combined}"""
        
        try:
            # Generate response using Ollama
            response = ollama.generate(
                model="llama2", # Using llama2 model
                prompt=prompt,
                options={
                    "temperature": 0.3, # Low temperature for more deterministic, factual output
                    "num_ctx": 2048,   # Context window size
                    "top_k": 40        # Top-k sampling
                }
            )
            # Clean and format the LLM's raw response
            return self._clean_summary(response['response'])
        except Exception as e:
            print(f"Error generating LLM summary for {sentiment} reviews: {e}")
            # Fallback to an NLP-based feature extraction if LLM call fails
            return self._fallback_feature_summary(texts, sentiment)

    def _fallback_feature_summary(self, texts, sentiment):
        """
        A fallback method to extract simple feature-opinion pairs using TF-IDF
        and basic keyword extraction if the LLM summary generation fails.

        Args:
            texts (list[str]): List of review texts.
            sentiment (str): Sentiment type ('positive' or 'negative').

        Returns:
            str: A formatted string of bulleted feature-opinion pairs.
        """
        # Extract prominent noun phrases as potential game features
        noun_phrases = self._extract_game_features(texts)
        
        feature_opinions = []
        # For each of the top 5 extracted phrases, try to find a common opinion
        for phrase in noun_phrases[:5]:
            opinion = self._extract_feature_opinion(texts, phrase)
            if opinion:
                feature_opinions.append(f"- {phrase}: {opinion}")
        
        if feature_opinions:
            return "\n".join(feature_opinions)
        return f"Couldn't extract specific {sentiment} features using fallback."

    def _extract_game_features(self, texts):
        """
        Extracts top game-specific noun phrases (1-gram to 3-gram) from texts
        using TF-IDF vectorization to identify important terms.

        Args:
            texts (list[str]): List of review texts.

        Returns:
            list[str]: A list of the most important feature terms.
        """
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), # Consider single words, two-word phrases, and three-word phrases
            max_features=50,    # Limit to top 50 features
            stop_words=list(self.stopwords) # Remove common words
        )
        # Fit and transform the texts to get TF-IDF scores
        tfidf = vectorizer.fit_transform(texts)
        # Get the feature names (words/phrases)
        return vectorizer.get_feature_names_out()

    def _extract_feature_opinion(self, texts, feature):
        """
        Attempts to find the most common opinion word associated with a specific feature
        within the given review texts. This is a simple proximity-based extraction.

        Args:
            texts (list[str]): List of review texts.
            feature (str): The specific game feature to find opinions for.

        Returns:
            str: The most common opinion word, or "is notable" if no specific
                 opinion word is found nearby.
        """
        opinion_words = []
        for text in texts:
            if feature in text:
                words = text.split()
                try:
                    idx = words.index(feature)
                    # Look for opinion words within a small window around the feature
                    for i in range(max(0, idx-2), min(len(words), idx+3)):
                        # Exclude the feature itself and very short words
                        if words[i] != feature and len(words[i]) > 4:
                            opinion_words.append(words[i])
                except ValueError:
                    # Feature might be part of a multi-word token after splitting
                    # This simple approach might miss it if `feature` is "gameplay"
                    # but `words` contains "game", "play".
                    pass # Continue to next text if feature not found directly as a word

        if opinion_words:
            # Return the most frequent opinion word
            return Counter(opinion_words).most_common(1)[0][0]
        return "is notable" # Default opinion if none found

    def _clean_summary(self, text):
        """
        Cleans and formats the raw text output from the LLM, ensuring it adheres
        to the desired single-sentence bullet point structure.
        This involves aggressive pattern matching to remove unwanted lines
        (e.g., instructions, bad examples, meta-commentary) and refining
        the remaining relevant feature sentences.

        Args:
            text (str): The raw text response from the LLM.

        Returns:
            str: A multi-line string with each cleaned feature as a bullet point.
        """
        lines = text.split('\n')
        final_cleaned_features = []
        
        # Define patterns for lines that should be discarded from the LLM output
        discard_patterns = [
            # General introductory/concluding phrases, headers, or summaries from LLM
            r"^\s*(here are|in summary|the following|based on|players often mention|these reviews highlight|note:|summary of (positive|negative) reviews|ok(ay)?,\s*here are|this game has been praised for|this game has received criticism for|the most common (positive|negative) feedback involves)",
            r"^\s*\d+-\d+\s+specific features", # LLM might output "3-5 specific features"
            r"^\s*specific features players mention",
            r"^\s*this game's reviews highlight",
            
            # Lines containing review counts or similar numerical annotations
            r".*\(mentioned (in )?\d+ (reviews|times)\)",
            r".*\(\d+\)$", # Catches "(3)"
            r".*\[\d+\s*(review(s)?|time(s)?)\]$",
            
            # Lines that are direct quotes or examples (as per prompt instructions)
            r"^\s*[-*\•]?\s*['\"].*['\"]\s*(\(review \d+\))?$", # Lines that are just quotes
            r"^\s*[-*\•]?\s*\[quote\]:.+",
            r"^\s*[-*\•]?\s*example(s)?:",
            r"^\s*good examples:", # Lines from the "Good Examples" section of the prompt
            r"^\s*bad examples:",  # Lines from the "Bad Examples" section of the prompt
            r"^\s*\(review \d+\)$", # Standalone review references
            
            # LLM's own formatting rules or placeholder content that is not a feature
            r"^\s*\[feature name\][:]?\s*\[concise opinion.*\]", # Placeholder structure
            r"^\s*rules:$",       # The "Rules:" header from the prompt
            r"^\s*positive feedback:", # Internal LLM headings
            r"^\s*negative feedback:", # Internal LLM headings
            r"^\s*(feature|opinion):.*", # Remove lines that explicitly state "Feature:" or "Opinion:"
            r"^\s*\"[^\"]*\"$",   # Lines that are just quotes (double quotes)
            r"^\s*'.*'$"          # Lines that are just quotes (single quotes)
        ]

        # Regex to identify common bullet point prefixes (hyphen, asterisk, bullet symbol, numbered list)
        bullet_prefix_pattern = re.compile(r"^\s*([-\*\•]|\d+\.)\s*") 

        for line in lines:
            original_line_stripped = line.strip()
            if not original_line_stripped:
                continue # Skip empty lines

            # Create a version of the line for pattern checking, stripping bullet and converting to lower
            line_content_for_pattern_check = bullet_prefix_pattern.sub("", original_line_stripped).strip().lower()
            
            should_discard = False
            # Check if the line matches any of the discard patterns
            for pattern in discard_patterns:
                if re.search(pattern, original_line_stripped.lower()) or \
                   re.search(pattern, line_content_for_pattern_check):
                    should_discard = True
                    break
            
            if should_discard:
                continue # Skip this line if it matches a discard pattern

            # Remove bullet prefixes from the line content
            cleaned_line_content = bullet_prefix_pattern.sub("", original_line_stripped).strip()
            
            # Aggressively remove any trailing counts like "(3)", "(X reviews)", "[X times]"
            # This regex targets various forms of numerical annotations at the end of a line.
            cleaned_line_content = re.sub(r"\s*(\(|\[)\s*(\d+\s*(reviews?|times?|)\s*)?\d+(\s*(reviews?|times?|)\s*)?(\)|\])$", "", cleaned_line_content).strip()
            
            # Remove any leading or trailing quotes that might have slipped through
            cleaned_line_content = re.sub(r"^[\"']|[\"']$", "", cleaned_line_content).strip()

            # Final validation: Ensure it's a substantive line, contains alphabetic characters,
            # doesn't start with "feature:" or "opinion:", and ends with punctuation.
            if len(cleaned_line_content) > 10 and any(c.isalpha() for c in cleaned_line_content) \
               and not cleaned_line_content.lower().startswith(("feature:", "opinion:")) \
               and re.search(r"[.!?]$", cleaned_line_content): # Ensure it ends with punctuation
                final_cleaned_features.append(cleaned_line_content)

        # Remove duplicate lines while preserving order (using dict.fromkeys)
        final_cleaned_features = list(dict.fromkeys(final_cleaned_features))

        # Format remaining cleaned features as bullet points
        return "\n".join(f"- {feature}" for feature in final_cleaned_features if feature)

    def _fallback_summary(self, texts, sentiment):
        """
        An alternative fallback summary generation method using TF-IDF to identify
        common phrases if the LLM or primary fallback methods fail.

        Args:
            texts (list[str]): List of review texts.
            sentiment (str): Sentiment type ('positive' or 'negative').

        Returns:
            str: A formatted string of common phrases.
        """
        vectorizer = TfidfVectorizer(
            stop_words=list(self.stopwords),
            max_features=100,
            ngram_range=(1, 3)
        )
        try:
            tfidf = vectorizer.fit_transform(texts)
            features = vectorizer.get_feature_names_out()
            scores = np.asarray(tfidf.sum(axis=0)).ravel()
            
            # Get indices of top 5 features by their TF-IDF scores
            top_indices = scores.argsort()[-5:][::-1]
            phrases = [features[i] for i in top_indices]
            
            summary = f"Common {sentiment} phrases:\n"
            summary += "\n".join(f"- Many mention '{phrase}'" for phrase in phrases)
            return summary
        except Exception as e:
            print(f"Error in fallback summary generation for {sentiment} aspects: {e}")
            return f"Could not analyze {sentiment} aspects due to an error."

    def _classify_reviews(self, reviews):
        """
        Classifies a list of raw reviews into positive and negative categories
        by lemmatizing the text and then using an LLM for sentiment classification.
        Processes reviews in batches for efficiency.

        Args:
            reviews (list[dict]): A list of review dictionaries (e.g., from Steam API).

        Returns:
            tuple[list[str], list[str]]: A tuple containing two lists:
                                         (positive_review_texts, negative_review_texts).
        """
        positive = []
        negative = []
        
        batch_size = 10 # Process reviews in batches to manage LLM calls
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            for review in batch:
                # Lemmatize the review text before classification
                text = self._lemmatize_text(review['review'])
                sentiment = self._classify_sentiment(text)
                if sentiment == 'positive':
                    positive.append(text)
                elif sentiment == 'negative':
                    negative.append(text)
        
        return positive, negative

    def _classify_sentiment(self, text):
        """
        Classifies the sentiment of a single text using the Ollama language model.
        The LLM is prompted to return 'positive' or 'negative' directly.

        Args:
            text (str): The review text to classify.

        Returns:
            str: 'positive', 'negative', or 'neutral' if classification fails.
        """
        # Prompt the LLM for a single-word sentiment classification
        prompt = f"""Classify this game review sentiment:
                   Text: "{text[:500]}" # Limit text length for prompt
                   Options: [positive, negative]
                   Respond with ONLY one word:"""
        try:
            response = ollama.generate(
                model="llama2",
                prompt=prompt,
                options={"temperature": 0.0} # Zero temperature for deterministic classification
            )
            res = response['response'].strip().lower()
            # Return the classified sentiment, default to 'neutral' if unexpected output
            return res if res in ['positive', 'negative'] else 'neutral'
        except Exception as e:
            print(f"Error classifying sentiment for text '{text[:50]}...': {e}")
            return 'neutral' # Return neutral on error

    def _fetch_reviews(self, appid, max_reviews):
        """
        Fetches recent English reviews for a given Steam game from the Steam API.

        Args:
            appid (str): The Steam application ID.
            max_reviews (int): The maximum number of reviews to retrieve.

        Returns:
            list[dict]: A list of review dictionaries. Returns an empty list on error.
        """
        url = f"https://store.steampowered.com/appreviews/{appid}"
        params = {
            "json": 1,           # Request JSON response
            "filter": "recent",  # Filter by recent reviews
            "language": "english", # Only English reviews
            "day_range": 365,    # Reviews from the last year
            "num_per_page": min(100, max_reviews), # Max 100 reviews per API call, capped by max_reviews
            "purchase_type": "all" # Include all purchase types
        }
        
        try:
            response = requests.get(url, params=params, timeout=15) # Set a timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            return data.get('reviews', [])[:max_reviews] # Return reviews, truncated to max_reviews
        except requests.exceptions.RequestException as e:
            print(f"Fetch reviews error during API request for appid {appid}: {e}")
        except ValueError as e:
            print(f"Fetch reviews error: Could not decode JSON response for appid {appid}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while fetching reviews for appid {appid}: {e}")
        return []

    def _create_wordcloud(self, texts, colormap, sentiment_type):
        """
        Generates a word cloud image from a list of texts, focusing on words
        relevant to the specified sentiment type.

        Args:
            texts (list[str]): List of review texts.
            colormap (str): Matplotlib colormap to use (e.g., 'Greens', 'Reds').
            sentiment_type (str): 'positive' or 'negative' to filter keywords.

        Returns:
            wordcloud.WordCloud: A WordCloud object ready for display, or None if
                                 not enough texts or an error occurs.
        """
        if not texts or len(texts) < 3:
            return None # Need at least 3 texts to create a meaningful word cloud
            
        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(self.stopwords),
                max_features=200,    # Consider up to 200 most important terms
                ngram_range=(1, 2)   # Include single words and two-word phrases
            )
            tfidf = vectorizer.fit_transform(texts)
            words = vectorizer.get_feature_names_out()
            weights = np.asarray(tfidf.sum(axis=0)).ravel().tolist()
            
            filtered_words_and_weights = {}
            # Define specific keywords for positive and negative sentiments
            positive_keywords = ["fun", "great", "enjoy", "love", "amazing", "addictive", "beautiful", "good", "excellent", "best", "solid", "perfect", "masterpiece", "charming", "impressive", "smooth", "polished", "rich", "deep", "engaging", "vast", "stable", "addictive"]
            negative_keywords = ["bug", "glitch", "crash", "broken", "bad", "terrible", "frustrating", "lag", "unplayable", "issues", "worst", "problem", "poor", "unstable", "disappointing", "boring", "mess", "clunky", "repetitive", "empty", "shoddy", "shallow", "missing", "woke", "cringe"]

            # Filter words to emphasize sentiment-specific terms
            for word, weight in zip(words, weights):
                if sentiment_type == 'positive' and word in positive_keywords:
                    filtered_words_and_weights[word] = weight * 2 # Boost weight for direct sentiment keywords
                elif sentiment_type == 'negative' and word in negative_keywords:
                    filtered_words_and_weights[word] = weight * 2 # Boost weight for direct sentiment keywords
                elif word not in filtered_words_and_weights and weight > 0.05: # Include other words above a certain TF-IDF threshold
                    filtered_words_and_weights[word] = weight

            if not filtered_words_and_weights:
                return None # No significant words found after filtering

            # Generate and return the word cloud
            return WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=colormap,       # Apply specified color scheme
                max_words=100,           # Max words to display in the cloud
                collocations=False       # Do not include collocations (phrases)
            ).generate_from_frequencies(filtered_words_and_weights)
        except Exception as e:
            print(f"Wordcloud generation error: {e}")
            return None

    def _lemmatize_text(self, text):
        """
        Normalizes text by converting words to their base form (lemma) using NLTK's WordNetLemmatizer.

        Args:
            text (str): The input text.

        Returns:
            str: The lemmatized text.
        """
        try:
            return ' '.join([self.wnl.lemmatize(word) for word in nltk.word_tokenize(text)])
        except Exception as e:
            print(f"Error during text lemmatization: {e}")
            return text # Return original text on error