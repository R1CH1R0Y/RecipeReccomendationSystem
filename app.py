import pickle
import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your pre-trained models and data
data = pd.read_csv('preprocessed_recipes.csv')

# Ensure NaN values in 'clean_ingredients' are replaced with an empty string
data['clean_ingredients'] = data['clean_ingredients'].fillna('')

# Load the saved TF-IDF vectorizer
tfidf = pickle.load(open('model1.pkl', 'rb'))

# Create the TF-IDF matrix from the preprocessed data
tfidf_matrix = tfidf.transform(data['clean_ingredients'])

# Function to clean ingredients (similar to your original version)
def clean_ingredients(ingredients):
    if isinstance(ingredients, str):
        ingredients = ingredients.strip("[]").replace("'", "").split(",")
        ingredients = [ingredient.strip().lower() for ingredient in ingredients]
        return ', '.join(ingredients)
    return ''

# Recommendation function based on cosine similarity
def recommend_recipes(input_ingredients, top_n=5):
    # Preprocess the user's input ingredients to match the format of the dataset
    input_ingredients = clean_ingredients(str(input_ingredients))

    # Vectorize the user's input ingredients
    input_tfidf = tfidf.transform([input_ingredients])

    # Compute cosine similarity between the input and the recipes in the dataset
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()

    # Get the top_n most similar recipes
    top_n_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Retrieve the top recipes
    recommended_recipes = data.iloc[top_n_indices][['name', 'clean_ingredients', 'minutes', 'steps']]
    return recommended_recipes

# Streamlit UI setup
st.header('Recipe Recommender System')

# User input for ingredients (no default text)
ingredients_input = st.text_input("Enter ingredients separated by commas", "")
ingredients_list = [ingredient.strip() for ingredient in ingredients_input.split(",") if ingredient.strip()]

recipe_number = 0
# Show recommendations when button is clicked
if st.button('Show Recommendations'):
    if not ingredients_list:  # Check if no ingredients were input
        st.warning("Please enter at least one ingredient.")
    else:
        
        # Get recommendations based on input ingredients
        recommended_recipes = recommend_recipes(ingredients_list)

        # Display recommendations with a heading
        if recommended_recipes.empty:
            st.write("No match found")
        else:
            # Add a heading for the recommendations section
            st.markdown("## **Top Recommendations**")

            # Create the recommendation table row by row
            for i, row in recommended_recipes.iterrows():
                # Recipe numbering starts from 1
                recipe_number = recipe_number + 1
                
                # Correctly capitalize the recipe name and format the title
                recipe_name = row['name'].capitalize()  # Capitalize the recipe name

                # Display the recipe title
                st.markdown("### **Recipe {}: {}**".format(recipe_number, recipe_name))
                
                # Display the structured details of the recipe in a table format
                ingredients_list = row['clean_ingredients'].split(", ")  # Split the ingredients for point-wise display
                
                # Capitalize the first letter of each ingredient
                formatted_ingredients = '<br>'.join(["- {}".format(ingredient.strip().capitalize()) for ingredient in ingredients_list])

                # Parse the steps from the CSV and format them similarly
                steps = row['steps']
                try:
                    steps_list = ast.literal_eval(steps)
                    # Clean the step text and capitalize the first letter of each step
                    formatted_steps = '<br>'.join(["{}. {}".format(j + 1, step.strip().capitalize().replace("'", '')) for j, step in enumerate(steps_list)])
                except (ValueError, SyntaxError):
                    formatted_steps = "No steps available for this recommendation."
                
                # Display the information in a structured table
                st.markdown("""
                <table>
                    <tr>
                        <th>Total Time Required</th>
                        <th>Ingredients Required</th>
                        <th>Steps</th>
                    </tr>
                    <tr>
                        <td>{} minutes</td>
                        <td>{}</td>
                        <td>{}</td>
                    </tr>
                </table>
                """.format(row['minutes'], formatted_ingredients, formatted_steps), unsafe_allow_html=True)