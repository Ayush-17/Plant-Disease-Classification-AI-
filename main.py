import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import os
import json
import time

# Function to get cure and prevention information from Groq API
def get_disease_info_from_groq(disease_name):
    # Check if API key is available
    if not st.session_state.groq_api_key:
        st.warning("Please enter a Groq API key in the sidebar to get real-time disease information.")
        return get_disease_cure_from_local(disease_name)
        
    # Clean up disease name for the prompt
    clean_disease_name = disease_name.replace('___', ' ').replace('_', ' ')
    
    try:
        # Prepare the prompt for Groq API with explicit JSON formatting instructions
        prompt = f"""Provide detailed information about the disease '{clean_disease_name}' in plants.
Include:
1. Treatment and cure methods
2. Prevention measures

Format your response ONLY as a JSON object with exactly these two keys: 'cure' and 'prevention'.
For both cure and prevention, provide the information in numbered bullet points (1., 2., 3., etc).

Example format:
{{
  "cure": "1. First cure method\\n2. Second cure method\\n3. Third cure method",
  "prevention": "1. First prevention method\\n2. Second prevention method\\n3. Third prevention method"
}}

IMPORTANT: Ensure your response is a valid JSON object and nothing else."""
        
        # Call Groq API
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.groq_api_key}"
        }
        payload = {
            "model": "llama3-8b-8192",  # Using Llama 3 8B model
            "messages": [
                {"role": "system", "content": "You are an expert in plant pathology and agricultural science. You provide information about plant diseases in properly formatted JSON only. Never include markdown, backticks, or anything outside the JSON object."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,  # Lowered temperature for more consistent output
            "max_tokens": 800
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        response_data = response.json()
        
        # Extract content from response
        content = response_data["choices"][0]["message"]["content"]
        
        # Clean the content - remove markdown code blocks if present
        cleaned_content = content
        if "```json" in content:
            # Remove markdown code blocks
            cleaned_content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            # Remove any code blocks
            cleaned_content = content.split("```")[1].split("```")[0].strip()
            
        # Parse JSON response with improved error handling
        try:
            # First try to parse directly
            result = json.loads(cleaned_content)
            
            # Ensure the response has the expected structure
            if 'cure' not in result or 'prevention' not in result:
                # If keys are missing, add them with default values
                if 'cure' not in result:
                    result['cure'] = "Specific cure information not available from API."
                if 'prevention' not in result:
                    result['prevention'] = "Prevention information not available from API."
                    
            return result
            
        except json.JSONDecodeError as e:
            # If direct JSON parsing fails, try fixing common JSON formatting issues
            st.warning(f"API response was not in proper JSON format: {str(e)}. Attempting to fix...")
            
            try:
                # Try to fix common JSON errors
                import re
                
                # Replace single quotes with double quotes (common error)
                fixed_content = cleaned_content.replace("'", '"')
                
                # Fix missing quotes around keys
                fixed_content = re.sub(r'(\s*)(\w+)(\s*):([^"])', r'\1"\2"\3:\4', fixed_content)
                
                # Fix trailing commas in JSON objects
                fixed_content = re.sub(r',(\s*})', r'\1', fixed_content)
                
                # Add missing quotes around string values
                fixed_content = re.sub(r':\s*([^"{}\[\],\d][^,{}\[\]]*?)([,}])', r': "\1"\2', fixed_content)
                
                # Ensure proper line breaks are escaped
                fixed_content = fixed_content.replace('\n', '\\n')
                
                # Try to parse the fixed JSON
                result = json.loads(fixed_content)
                
                # Ensure the response has the expected structure
                if 'cure' not in result or 'prevention' not in result:
                    if 'cure' not in result:
                        result['cure'] = "Specific cure information not available from API."
                    if 'prevention' not in result:
                        result['prevention'] = "Prevention information not available from API."
                        
                return result
            except json.JSONDecodeError:
                # If fixing common errors didn't work, try more aggressive extraction
                st.warning("Couldn't fix JSON format. Using regex extraction...")
            
            # Try to find the JSON object within the text
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, cleaned_content, re.DOTALL)
            
            if match:
                try:
                    # Try parsing the extracted JSON
                    json_str = match.group(0)
                    result = json.loads(json_str)
                    
                    # Ensure the response has the expected structure
                    if 'cure' not in result or 'prevention' not in result:
                        if 'cure' not in result:
                            result['cure'] = "Specific cure information not available from API."
                        if 'prevention' not in result:
                            result['prevention'] = "Prevention information not available from API."
                            
                    return result
                except:
                    st.warning("Regex extraction failed. Using text extraction method.")
            
            # Extract JSON-like key-value pairs if we can find them
            try:
                st.warning("Attempting to extract key-value pairs directly...")
                # Look for cure and prevention sections with a more lenient approach
                cure_pattern = r'"cure"\s*:\s*"([^"]*)"'
                prevention_pattern = r'"prevention"\s*:\s*"([^"]*)"'
                
                cure_match = re.search(cure_pattern, cleaned_content, re.DOTALL)
                prevention_match = re.search(prevention_pattern, cleaned_content, re.DOTALL)
                
                if cure_match and prevention_match:
                    return {
                        'cure': cure_match.group(1).replace('\\n', '\n'),
                        'prevention': prevention_match.group(1).replace('\\n', '\n')
                    }
            except:
                st.warning("Key-value extraction failed. Using fallback text extraction.")
            
            # Text extraction as a last resort
            cure_info = "Specific cure information not available from API."
            prevention_info = "Prevention information not available from API."
            
            # Look for cure/treatment and prevention information using pattern matching
            cure_patterns = ["cure", "treatment", "management", "control"]
            prev_patterns = ["prevention", "preventive", "avoid", "reducing risk"]
            
            # Find cure information
            for pattern in cure_patterns:
                if pattern in cleaned_content.lower():
                    # Find pattern and extract surrounding text
                    pattern_idx = cleaned_content.lower().find(pattern)
                    if pattern_idx > -1:
                        # Find end of this section - look for prevention keywords or double newlines
                        end_idx = len(cleaned_content)
                        for prev_pattern in prev_patterns:
                            prev_idx = cleaned_content.lower().find(prev_pattern, pattern_idx)
                            if prev_idx > -1 and prev_idx < end_idx:
                                end_idx = prev_idx
                        
                        # Extract and clean the cure information
                        cure_text = cleaned_content[pattern_idx:end_idx].strip()
                        if cure_text:
                            cure_info = cure_text
                            break
            
            # Find prevention information
            for pattern in prev_patterns:
                if pattern in cleaned_content.lower():
                    # Find pattern and extract text to the end
                    pattern_idx = cleaned_content.lower().find(pattern)
                    if pattern_idx > -1:
                        prevention_text = cleaned_content[pattern_idx:].strip()
                        if prevention_text:
                            prevention_info = prevention_text
                            break
            
            # Clean and format the extracted text
            def format_as_bullet_points(text):
                # Remove the keyword from the start
                for kw in cure_patterns + prev_patterns:
                    if text.lower().startswith(kw):
                        text = text[len(kw):].lstrip(': \n')
                
                # Split by potential bullet points or newlines
                lines = re.split(r'\n+|\d+\.|\*|\-', text)
                lines = [line.strip() for line in lines if line.strip()]
                
                # Format as numbered bullet points
                if lines:
                    return "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
                return text
            
            return {
                'cure': format_as_bullet_points(cure_info),
                'prevention': format_as_bullet_points(prevention_info)
            }
    
    except Exception as e:
        st.error(f"Error fetching data from Groq API: {str(e)}")
        # Fall back to local dictionary if API fails
        return get_disease_cure_from_local(disease_name)

# Function to get cure and prevention information from OpenAI API
def get_disease_info_from_openai(disease_name):
    # Check if API key is available
    if not st.session_state.openai_api_key:
        st.warning("Please enter an OpenAI API key in the sidebar to get real-time disease information.")
        return get_disease_cure_from_local(disease_name)
        
    # Clean up disease name for the prompt
    clean_disease_name = disease_name.replace('___', ' ').replace('_', ' ')
    
    try:
        # Prepare the prompt for OpenAI API
        prompt = f"Provide detailed information about the disease '{clean_disease_name}' in plants. Include:\n1. Treatment and cure methods\n2. Prevention measures\nFormat the response as JSON with 'cure' and 'prevention' keys. For both cure and prevention, provide the information in numbered bullet points (1., 2., 3., etc)."
        
        # Call OpenAI API
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.session_state.openai_api_key}"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert in plant pathology and agricultural science. Provide accurate, detailed information about plant diseases, their treatments, and prevention methods in JSON format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        response_data = response.json()
        
        # Extract content from response (OpenAI API response structure)
        content = response_data["choices"][0]["message"]["content"]
        
        # Parse JSON response
        try:
            # First try to parse directly
            result = json.loads(content)
            
            # Ensure the response has the expected structure
            if 'cure' not in result or 'prevention' not in result:
                # If keys are missing, add them with default values
                if 'cure' not in result:
                    result['cure'] = "Specific cure information not available from API."
                if 'prevention' not in result:
                    result['prevention'] = "Prevention information not available from API."
                    
            return result
            
        except json.JSONDecodeError:
            # If direct JSON parsing fails, try to extract JSON-like content
            st.warning("API response was not in proper JSON format. Using fallback method.")
            
            # Look for cure and prevention information in the text
            cure_info = "Specific cure information not available from API."
            prevention_info = "Prevention information not available from API."
            
            # Simple text parsing for cure and prevention
            if "cure" in content.lower() or "treatment" in content.lower():
                cure_start = max(content.lower().find("cure"), content.lower().find("treatment"))
                if cure_start > -1:
                    cure_end = content.find("\n\n", cure_start)
                    if cure_end > -1:
                        cure_info = content[cure_start:cure_end].strip()
                    else:
                        cure_info = content[cure_start:].strip()
            
            if "prevention" in content.lower():
                prev_start = content.lower().find("prevention")
                if prev_start > -1:
                    prev_end = content.find("\n\n", prev_start)
                    if prev_end > -1:
                        prevention_info = content[prev_start:prev_end].strip()
                    else:
                        prevention_info = content[prev_start:].strip()
            
            return {
                'cure': cure_info,
                'prevention': prevention_info
            }
    
    except Exception as e:
        st.error(f"Error fetching data from OpenAI API: {str(e)}")
        # Fall back to local dictionary if API fails
        return get_disease_cure_from_local(disease_name)

# Function to get cure and prevention information from local dictionary (fallback)
def get_disease_cure_from_local(disease_name):
    cure_info = {
        'Apple___Apple_scab': {
            'cure': '1. Apply fungicides containing captan or myclobutanil at first sign of infection\n2. Remove and destroy infected leaves and fruits\n3. Prune trees to improve air circulation\n4. Apply sulfur-based fungicides during growing season\n5. Use copper-based sprays in early spring\n6. Monitor and repeat treatments every 7-10 days during wet weather',
            'prevention': '1. Plant disease-resistant apple varieties\n2. Rake and remove fallen leaves in autumn\n3. Maintain good air circulation through proper pruning\n4. Apply preventive fungicide sprays in early spring\n5. Keep grass and weeds trimmed around trees\n6. Avoid overhead watering'
        },
        'Apple___Black_rot': {
            'cure': '1. Remove all mummified fruits from trees and ground\n2. Prune out dead or diseased wood and cankers\n3. Apply fungicides containing thiophanate-methyl\n4. Treat nearby cedar trees if present\n5. Apply copper-based fungicides during dormant season\n6. Monitor and maintain tree health with proper fertilization',
            'prevention': '1. Clean up all fallen fruit and leaves\n2. Prune trees for better air circulation\n3. Maintain tree vigor through proper fertilization\n4. Control insects that can spread the disease\n5. Remove infected plant material promptly\n6. Use disease-resistant varieties when possible'
        },
        'Apple___Cedar_apple_rust': {
            'cure': 'Spray with fungicides during the growing season. Remove galls from cedar trees.',
            'prevention': 'Plant resistant varieties. Remove nearby cedar trees if possible.'
        },
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
            'cure': 'Apply appropriate fungicides. Remove severely infected plants.',
            'prevention': 'Crop rotation. Tillage practices to reduce residue.'
        },
        'Corn_(maize)___Common_rust_': {
            'cure': 'Apply registered fungicides. Remove severely infected plants.',
            'prevention': 'Plant resistant hybrids. Avoid late plantings.'
        },
        'Corn_(maize)___Northern_Leaf_Blight': {
            'cure': 'Apply fungicides at first sign of disease. Remove infected debris.',
            'prevention': 'Plant resistant varieties. Rotate crops.'
        },
        'Grape___Black_rot': {
            'cure': '1. Apply fungicides containing myclobutanil or thiophanate-methyl\n2. Remove mummified berries and infected leaves\n3. Improve vineyard air circulation\n4. Apply copper-based sprays\n5. Prune out infected canes and clusters\n6. Monitor clusters regularly during ripening',
            'prevention': '1. Remove all mummified fruit in winter\n2. Prune for good air circulation\n3. Train vines for maximum sun exposure\n4. Clean up fallen debris regularly\n5. Apply dormant sprays in early spring\n6. Maintain proper vine nutrition'
        },
        'Tomato___Late_blight': {
            'cure': '1. Apply copper-based fungicides immediately upon detection\n2. Remove and destroy all infected plant parts\n3. Increase plant spacing to improve air flow\n4. Apply organic fungicides containing Bacillus subtilis\n5. Use protective fungicides in wet weather\n6. Monitor plants daily for new infections',
            'prevention': '1. Plant resistant varieties\n2. Avoid overhead watering\n3. Space plants properly for good air circulation\n4. Mulch around plants to prevent soil splash\n5. Remove volunteer tomato plants\n6. Rotate crops every 3-4 years'
        },
        'Potato___Early_blight': {
            'cure': '1. Apply fungicides containing chlorothalonil or copper\n2. Remove infected leaves and destroy them\n3. Improve air circulation between plants\n4. Apply neem oil as an organic alternative\n5. Maintain proper plant nutrition\n6. Monitor and treat regularly during growing season',
            'prevention': '1. Practice crop rotation\n2. Plant resistant varieties\n3. Maintain proper plant spacing\n4. Keep plants well-mulched\n5. Avoid overhead irrigation\n6. Remove plant debris after harvest'
        },
        'Tomato___Leaf_Mold': {
            'cure': '1. Apply fungicides containing chlorothalonil\n2. Remove and destroy infected leaves\n3. Reduce humidity around plants\n4. Improve greenhouse ventilation if indoor\n5. Apply copper-based fungicides\n6. Space plants for better air circulation',
            'prevention': '1. Use resistant varieties\n2. Maintain low humidity\n3. Avoid leaf wetness\n4. Space plants properly\n5. Provide good ventilation\n6. Remove infected plant debris'
        }
    }
    
    # If the disease is in our dictionary, return the information
    if disease_name in cure_info:
        return cure_info[disease_name]
    else:
        # Return a comprehensive response for diseases not in our dictionary
        return {
            'cure': """1. Apply appropriate fungicides or pesticides specific to the identified disease
2. Remove and destroy infected plant parts to prevent spread
3. Improve air circulation around plants by proper spacing and pruning
4. Adjust watering practices to avoid over-watering
5. Apply organic treatments like neem oil or copper-based solutions if applicable
6. Monitor plant response to treatments and adjust as needed
7. Consider soil treatment if the disease is soil-borne
Note: For best results, please consult with a local agricultural expert for specific treatment options.""",
            'prevention': """1. Choose disease-resistant plant varieties
2. Maintain proper plant spacing for good air circulation
3. Practice crop rotation to prevent disease buildup in soil
4. Keep garden tools clean and sanitized
5. Monitor plants regularly for early signs of disease
6. Maintain optimal soil pH and nutrition levels
7. Use mulch to prevent soil-splash onto leaves
8. Avoid overhead watering to reduce leaf wetness"""}

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

# Initialize session state for API keys
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""  # Default API key provided
    
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""  # Default Groq API key

if 'api_selection' not in st.session_state:
    st.session_state.api_selection = "groq"  # Default to Groq

#Sidebar
st.sidebar.title("Dashboard")

# API Settings in sidebar
st.sidebar.header("AI API Settings")

# API Selection
st.session_state.api_selection = st.sidebar.radio(
    "Select AI API Service:",
    ["groq", "openai"],
    index=0 if st.session_state.api_selection == "groq" else 1
)

# Show appropriate API key input based on selection
if st.session_state.api_selection == "openai":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", value=st.session_state.openai_api_key)
    # Update the session state when a new key is entered
    if api_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key
else:  # Groq is selected
    api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", value=st.session_state.groq_api_key)
    # Update the session state when a new key is entered
    if api_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key

app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        if test_image is not None:
            st.image(test_image, width=4, use_column_width=True)
        else:
            st.error("Please upload an image first")
    
    #Predict button
    if(st.button("Predict")):
        if test_image is not None:
            with st.spinner("Analyzing your plant image..."):
                st.snow()
                st.write("Our Prediction")
                try:
                    result_index = model_prediction(test_image)
                    
                    #Reading Labels
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                                  'Tomato___healthy']
                    
                    # Check if result_index is valid
                    if 0 <= result_index < len(class_name):
                        predicted_disease = class_name[result_index]
                        st.success("Model is Predicting it's a {}".format(predicted_disease))
                        
                        # Store the predicted disease name in session state for use by the Show Cure button
                        st.session_state.predicted_disease = predicted_disease
                        
                        # Create a button for showing cure and prevention info
                        st.session_state.show_cure_button = True
                    else:
                        st.error("Invalid prediction result. Please try another image.")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please try uploading a clearer image or a different plant leaf.")
        else:
            st.error("Please upload an image first")

    # Main function to get disease cure and prevention info
    def get_disease_cure(disease_name):
        """
        Main function to get disease cure information, with retry mechanism and fallback
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                with st.spinner(f"Fetching information about {disease_name.replace('___', ' ')}... Attempt {attempt+1}/{max_retries}"):
                    # Try to get information from selected API
                    if st.session_state.api_selection == "openai":
                        info = get_disease_info_from_openai(disease_name)
                    else:  # Use Groq API
                        info = get_disease_info_from_groq(disease_name)
                    return info
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    st.error(f"All {max_retries} attempts failed. Using local database.")
                    # Fallback to local database
                    return get_disease_cure_from_local(disease_name)
    
    # Show Cure and Prevention button if prediction was made
    if hasattr(st.session_state, 'show_cure_button') and st.session_state.show_cure_button:
        if st.button("Show Cure and Prevention"):
            if st.session_state.api_selection == "openai" and not st.session_state.openai_api_key:
                st.warning("Please enter an OpenAI API key in the sidebar to get detailed information.")
            elif st.session_state.api_selection == "groq" and not st.session_state.groq_api_key:
                st.warning("Please enter a Groq API key in the sidebar to get detailed information.")
            
            if hasattr(st.session_state, 'predicted_disease'):
                st.subheader("Cure and Prevention Information")
                
                try:
                    # Get cure information for the predicted disease with loading indicator
                    with st.spinner("Fetching treatment and prevention information..."):
                        disease_info = get_disease_cure(st.session_state.predicted_disease)
                    
                    # Display the cure and prevention information in an organized way
                    st.markdown("### Disease: {}".format(st.session_state.predicted_disease.replace('___', ' - ')))
                    
                    # Create two columns for Cure and Prevention
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Treatment/Cure:")
                        st.info(disease_info['cure'])
                        
                    with col2:
                        st.markdown("#### Prevention:")
                        st.info(disease_info['prevention'])
                    
                    # Additional general advice
                    st.markdown("### Additional Tips")
                    st.warning("""
                        * Always follow product labels when applying any treatments
                        * Consider integrated pest management (IPM) approaches
                        * Consult with local agricultural extension for specific recommendations
                        * Maintain general plant health through proper watering and nutrition
                    """)
                    
                    # Add a feedback section
                    st.markdown("### Was this information helpful?")
                    col1, col2, col3 = st.columns([1,1,2])
                    with col1:
                        if st.button("👍 Yes"):
                            st.success("Thank you for your feedback!")
                    with col2:
                        if st.button("👎 No"):
                            st.info("We'll work on improving our recommendations.")
                            
                except Exception as e:
                    st.error(f"Error retrieving information: {str(e)}")
                    st.info("Please try again or check your internet connection.")
            else:
                st.warning("Please predict a disease first")

